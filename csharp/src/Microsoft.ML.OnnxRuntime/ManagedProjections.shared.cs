// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class helps to feed the NamedOnnxValue as inference input.
    /// It projects managed classes to OrtValues so they can be consumed
    /// by the native onnxruntime library. if possible, it will avoid copying data.
    /// The NamedOnnxValue can be a tensor, sequence or map.
    /// For recursive structures, create nested NamedOnnxValue instances.
    /// For example, a sequence instance would contain a list of NamedOnnxValue instances
    /// that in turn may represent tensors or other ONNX values.
    /// </summary>
    internal class ManagedOnnxType : IDisposable
    {
        readonly DisposableList<IDisposable> _disposables;
        readonly OrtValue _ortValue;
        bool _disposed = false;

        /// <summary>
        /// Provides access to non-owning instance of OrtValue
        /// </summary>
        /// <value>Provides access to the OrtValue to be used as input</value>
        internal OrtValue Value { get { return new OrtValue(_ortValue.Handle, false); } }

        /// <summary>
        /// Constructor to create an input OrtValue projection from managed data
        /// </summary>
        /// <param name="namedOnnxValue"></param>
        /// <param name="metadata"></param>
        /// <exception cref="OnnxRuntimeException"></exception>
        internal ManagedOnnxType(NamedOnnxValue namedOnnxValue, NodeMetadata metadata)
        {
            if (namedOnnxValue.ValueType != metadata.OnnxValueType)
            {
                throw new OnnxRuntimeException(ErrorCode.RuntimeException,
                    $"NamedOnnxValue: {namedOnnxValue.Name} has value type: {namedOnnxValue.ValueType} expected: {metadata.OnnxValueType}");
            }

            int requiredCapacity = 32;
            var disposables = new DisposableList<IDisposable>(requiredCapacity);
            try
            {
                _ortValue = CreateDispatchProjection(namedOnnxValue, metadata, disposables);
            }
            catch (Exception)
            {
                disposables.Dispose();
                throw;
            }
            _disposables = disposables;
        }

        /// <summary>
        /// Dispatches the creation of the projection
        /// </summary>
        /// <param name="namedOnnxValue"></param>
        /// <param name="metadata"></param>
        /// <param name="disposables"></param>
        /// <returns></returns>
        private OrtValue CreateDispatchProjection(NamedOnnxValue namedOnnxValue, NodeMetadata metadata, DisposableList<IDisposable> disposables)
        {
            OrtValue result;
            switch (namedOnnxValue.ValueType)
            {
                case OnnxValueType.ONNX_TYPE_TENSOR:
                    result = CreateTensorProjection(namedOnnxValue, metadata, disposables);
                    break;
                case OnnxValueType.ONNX_TYPE_SEQUENCE:
                    result = CreateSequenceProjection(namedOnnxValue, metadata, disposables);
                    break;
                case OnnxValueType.ONNX_TYPE_MAP:
                    result = CreateMapProjection(namedOnnxValue, metadata, disposables);
                    break;
                case OnnxValueType.ONNX_TYPE_OPTIONAL:
                    {
                        var optMeta = metadata.AsOptionalMetadata();
                        Debug.Assert(optMeta != null);
                        result = CreateDispatchProjection(namedOnnxValue, optMeta.ElementMeta, disposables);
                    }
                    break;
                default:
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument, "ManagedOnnxType can only project tensors, sequences, maps and optional types");
            }
            return result;
        }

        /// <summary>
        /// The function creates OrtValue objects for each element of the sequence
        /// and then creates an OrtValue for the whole sequence.
        /// </summary>
        /// <param name="namedOnnxValue">NamedOnnxValue containing a IEnumeralbe<NameOnnValue></param>
        /// <param name="metadata">sequence metadata</param>
        /// <param name="disposables">cleanup list</param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private OrtValue CreateSequenceProjection(NamedOnnxValue namedOnnxValue, NodeMetadata metadata, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            var elementMeta = metadata.AsSequenceMetadata().ElementMeta;
            var elementOnnxValue = elementMeta.OnnxValueType;
            var seqContainer = namedOnnxValue.AsEnumerable<NamedOnnxValue>();

            if (seqContainer is null)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                               $"NamedOnnxValue: {namedOnnxValue.Name} sequence does not contain NamedOnnxValue elements");
            }

            // Record all the ortValues belonging to the sequence locally
            var sequenceOrtValues = new List<OrtValue>();
            foreach (var element in seqContainer)
            {
                if (elementOnnxValue != element.ValueType)
                {
                    throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                        $"NamedOnnxValue: {namedOnnxValue.Name} sequence element expected to be {elementOnnxValue}, received {element.ValueType}");
                }

                sequenceOrtValues.Add(CreateDispatchProjection(element, elementMeta, disposables));
            }

            IntPtr[] ortValHandles = new IntPtr[sequenceOrtValues.Count];
            for (int i = 0; i < sequenceOrtValues.Count; i++)
            {
                ortValHandles[i] = sequenceOrtValues[i].Handle;
            }

            using (var memHandle = new Memory<IntPtr>(ortValHandles).Pin())
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateValue(ortValHandles,
                    (UIntPtr)sequenceOrtValues.Count, (IntPtr)OnnxValueType.ONNX_TYPE_SEQUENCE, out IntPtr sequenceHandle));
                result = new OrtValue(sequenceHandle);
                disposables.Add(result);
            }

            return result;
        }

        /// <summary>
        /// Creates map projection. Since we support only primitive types in maps
        /// we map two tensors (keys and values)
        /// </summary>
        /// <param name="node"></param>
        /// <param name="elementMeta"></param>
        /// <param name="disposables"></param>
        /// <returns>OrtValue</returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private OrtValue CreateMapProjection(NamedOnnxValue node, NodeMetadata elementMeta, DisposableList<IDisposable> disposables)
        {
            OrtValue result = null;
            var mapMeta = elementMeta.AsMapMetadata();
            Debug.Assert(mapMeta != null);
            // Maps currently support only primitive types expressed as two parallel tensors and not nested Sequences or Maps

            var mapValuesMeta = mapMeta.ValueMetadata;
            if (mapValuesMeta.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Node: {node.Name} onnxruntime only supports maps with primitive types values");
            }


            var keys = node.GetDictionaryKeys();
            var ortValueKeys = OrtValue.CreateFromTensorObject(keys,
                    out MemoryHandle? memoryHandleKeys, out TensorElementType elementTypeKeys);
            disposables.Add(ortValueKeys);

            if (memoryHandleKeys.HasValue)
            {
                disposables.Add(memoryHandleKeys);
            }

            if (elementTypeKeys != mapMeta.KeyDataType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                       $"Map key data type supplied: {elementTypeKeys} metadata expected: {mapMeta.KeyDataType}");
            }

            var values = node.GetDictionaryValues();
            var ortValueValues = OrtValue.CreateFromTensorObject(values,
                    out MemoryHandle? memoryHandleValues, out TensorElementType elementTypeValues);

            disposables.Add(ortValueValues);
            if (memoryHandleValues.HasValue)
            {
                disposables.Add(memoryHandleValues);
            }

            if (elementTypeValues != mapValuesMeta.ElementDataType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                                                          $"Map value data type supplied: {elementTypeValues} metadata expected: {mapValuesMeta.ElementDataType}");
            }

            // Create Map OrtValue
            IntPtr[] ortValHandles = new IntPtr[2] { ortValueKeys.Handle, ortValueValues.Handle };
            using (var pinnedHandles = new Memory<IntPtr>(ortValHandles).Pin())
            {
                NativeApiStatus.VerifySuccess(NativeMethods.OrtCreateValue(ortValHandles, (UIntPtr)2,
                    (IntPtr)OnnxValueType.ONNX_TYPE_MAP, out IntPtr ortValueMap));
                result = new OrtValue(ortValueMap);
                disposables.Add(result);
            }
            return result;
        }


        /// <summary>
        /// This pins memory that is contained within DenseTensor.
        /// </summary>
        /// <param name="node">NodeOnnxValue containing DenseTensor</param>
        /// <param name="elementMeta"></param>
        /// <param name="disposables">cleanup list</param>
        /// <returns></returns>
        /// <exception cref="OnnxRuntimeException"></exception>
        private OrtValue CreateTensorProjection(NamedOnnxValue node, NodeMetadata elementMeta, DisposableList<IDisposable> disposables)
        {
            var ortValue = OrtValue.CreateFromTensorObject(node.Value,
                out MemoryHandle? memoryHandle, out TensorElementType elementType);
            disposables.Add(ortValue);

            if (memoryHandle.HasValue)
            {
                disposables.Add(memoryHandle);
            }

            if (elementType != elementMeta.ElementDataType)
            {
                throw new OnnxRuntimeException(ErrorCode.InvalidArgument,
                    $"Tensor element data type discovered: {elementType} metadata expected: {elementMeta.ElementDataType}");
            }

            return ortValue;
        }

        #region IDisposable
        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked by Dispose()</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }

            // dispose managed state (managed objects).
            if (disposing)
            {
                _disposables.Dispose();
            }
            _disposed = true;
        }


        public void Dispose()
        {
            Dispose(true);
        }

        #endregion IDisposable
    }
}

