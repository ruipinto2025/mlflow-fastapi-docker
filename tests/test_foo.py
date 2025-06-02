from mlflow_fastapi_docker.foo import foo


def test_foo():
    assert foo("foo") == "foo"
