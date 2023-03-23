// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// The class associates a name with an Object. Currently it supports Tensor<T>
    /// as possible objects. The name of the class is a misnomer, it does not hold any
    /// Onnx values.
    /// </summary>
    public class NamedOnnxValue
    {
        /// <summary>
        /// Managed Tensor, Dictionary or IList
        /// </summary>
        protected Object _value;
        /// <summary>
        /// Name of the instance, model input/output
        /// </summary>
        protected string _name;

        /// <summary>
        /// Constructs an instance of NamedOnnxValue and represents
        /// a model input to an inference session. It also represents a modle output
        /// when serves as a base for DisposablenamedOnnxvalue
        /// </summary>
        /// <param name="name">input/output name</param>
        /// <param name="value">Object that may be a tensor, Dictionary, IList</param>
        [Obsolete("This the constructor with valueType or static factory methods")]
        protected NamedOnnxValue(string name, Object value)
        {
            _name = name;
            _value = value;
            ValueType = OnnxValueType.ONNX_TYPE_UNKNOWN;
        }

        protected NamedOnnxValue(string name, Object value, OnnxValueType valueType, NodeMetadata metadata)
        {
            _name = name;
            _value = value;
            ValueType = valueType;
            Metadata = metadata;
        }

        /// <summary>
        /// Onnx Value Type if known. In general, NamedOnnxValue is able to contain
        /// arbitrary objects.
        /// </summary>
        public OnnxValueType ValueType { get; }

        /// <summary>
        /// Node metadata. Present when this is not a DisposableNamedOnnxValue
        /// </summary>
        public NodeMetadata Metadata { get; set; }

        /// <summary>
        /// This is a factory method that instantiates NamedOnnxValue
        /// and associated name with an instance of a Tensor<typeparamref name="T"/>
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="name">name</param>
        /// <param name="value">Tensor<typeparamref name="T"/></param>
        /// <param name="metadata">node metadata, necessary if this is used for input/output<typeparamref name="T"/></param>
        /// <returns></returns>
        public static NamedOnnxValue CreateFromTensor<T>(string name, Tensor<T> value, NodeMetadata metadata)
        {
            return new NamedOnnxValue(name, value, OnnxValueType.ONNX_TYPE_TENSOR, metadata);
        }

        /// <summary>
        /// This is a factory method that instantiates NamedOnnxValue.
        /// It would contain a sequence of elements
        /// </summary>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static NamedOnnxValue CreateFromSequence<T>(string name, IEnumerable<T> value, NodeMetadata metadata)
        {
            return new NamedOnnxValue(name, value, OnnxValueType.ONNX_TYPE_SEQUENCE, metadata);
        }

        /// <summary>
        /// This is a factory method that instantiates NamedOnnxValue.
        /// </summary>
        /// <typeparam name="K"></typeparam>
        /// <typeparam name="V"></typeparam>
        /// <param name="name"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        //public static NamedOnnxValue CreateFromMap<K, V>(string name, IDictionary<K, V> value)
        //{
        //    return new NamedOnnxValue(name, value, OnnxValueType.ONNX_TYPE_MAP);
        //}

        /// <summary>
        /// Exposes the name of the of the model input/output
        /// </summary>
        /// <value>name string</value>
        public string Name { get { return _name; } set { _name = value; } }
        /// <summary>
        /// Exposes the underlying managed object
        /// </summary>
        /// <value>object</value>
        public Object Value { get { return _value; } set { _value = value; } }

        /// <summary>
        /// Try-get value as a Tensor&lt;T&gt;.
        /// </summary>
        /// <typeparam name="T">Type</typeparam>
        /// <returns>Tensor object if contained value is a Tensor. Null otherwise</returns>
        public Tensor<T> AsTensor<T>()
        {
            return _value as Tensor<T>;  // will return null if not castable
        }

        /// <summary>
        /// Try-get value as an Enumerable&lt;T&gt;.
        /// </summary>
        /// <typeparam name="T">Type</typeparam>
        /// <returns>Enumerable object if contained value is a Enumerable. Null otherwise</returns>
        public IEnumerable<T> AsEnumerable<T>()
        {
            var x = _value as IEnumerable<T>;
            return x;
        }

        /// <summary>
        /// Try-get value as an Dictionary&lt;K,V&gt;.
        /// </summary>
        /// <typeparam name="K">Key type</typeparam>
        /// <typeparam name="V">Value type</typeparam>
        /// <returns>Dictionary object if contained value is a Dictionary. Null otherwise</returns>
        public IDictionary<K, V> AsDictionary<K, V>()
        {
            return _value as IDictionary<K, V>;
        }

        /// <summary>
        /// Pin the underlying memory and create an instance of OrtValue containing a tensor
        /// based on the pinned managed memory. The caller is responsible for Disposing
        /// both OrtValue and pinnedMemoryHandle
        /// </summary>
        /// <param name="pinnedMemoryHandle">dispose after returned OrtValus is disposed</param>
        /// <returns>an instance of OrtValue. The lifespan of OrtValue must overlap pinnedMemoryHandle</returns>
        internal virtual OrtValue ToOrtValue(out IDisposable memoryOwner)
        {
            var ortValue = OrtValue.CreateFromTensorObject(_value, out MemoryHandle? memoryHandle,
                out _);
            memoryOwner = memoryHandle;
            return ortValue;
        }

        // may expose different types of getters in future

    }
}
