def main():
    import tensorflow as tf
    tf.print("Hello, tensorflow!")
    tf.print(tf.__version__)
    print("Cuda:", tf.test.is_built_with_cuda())
    print("GPU:", tf.test.is_gpu_available())
    print(tf.config.list_logical_devices("GPU"))


if __name__ == "__main__":
    main()
