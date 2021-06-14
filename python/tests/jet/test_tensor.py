import pytest

import jet


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
class TestTensor:
    def test_default_constructor(self, dtype):
        """Tests that the default constructor is called."""
        tensor = jet.Tensor(dtype=dtype)
        assert tensor.indices == []
        assert tensor.shape == []
        assert tensor.data == [0]
        assert tensor.index_to_dimension_map == {}

    def test_shape_constructor(self, dtype):
        """Tests that the (shape) constructor is called."""
        tensor = jet.Tensor(shape=[1, 2, 3], dtype=dtype)
        assert tensor.indices == ["?a", "?b", "?c"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 0, 0, 0, 0, 0]
        assert tensor.index_to_dimension_map == {"?a": 1, "?b": 2, "?c": 3}

    def test_shape_index_constructor(self, dtype):
        """Tests that the (shape, indices) constructor is called."""
        tensor = jet.Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], dtype=dtype)
        assert tensor.indices == ["i", "j", "k"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 0, 0, 0, 0, 0]
        assert tensor.index_to_dimension_map == {"i": 1, "j": 2, "k": 3}

    def test_shape_index_data_constructor(self, dtype):
        """Tests that the (shape, indices, data) constructor is called."""
        tensor = jet.Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], data=range(6), dtype=dtype)
        assert tensor.indices == ["i", "j", "k"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 1, 2, 3, 4, 5]
        assert tensor.index_to_dimension_map == {"i": 1, "j": 2, "k": 3}

    def test_copy_constructor(self, dtype):
        """Tests that the copy constructor is called."""
        tensor_1 = jet.Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], data=range(6), dtype=dtype)
        tensor_2 = jet.Tensor(other=tensor_1, dtype=dtype)
        assert tensor_1 == tensor_2

    def test_set_shape(self, dtype):
        """Tests that the shape of a tensor can be modified."""
        tensor = jet.Tensor(shape=[1, 2, 3, 4], dtype=dtype)
        tensor.shape = [2, 6, 1, 1, 1]
        assert tensor.shape == [2, 6, 1, 1, 1]

    def test_getitem(self, dtype):
        """Tests that the data of a tensor can be randomly accessed using __getitem__."""
        tensor = jet.Tensor(shape=[3], indices=["i"], data=range(3), dtype=dtype)
        assert tensor[0] == 0
        assert tensor[1] == 1
        assert tensor[2] == 2

    def test_len(self, dtype):
        """Tests that the size of a tensor can be retrieved using __len__."""
        tensor = jet.Tensor(shape=[2, 3, 4], dtype=dtype)
        assert len(tensor) == 2 * 3 * 4

    def test_repr(self, dtype):
        """Tests that the string representation of a tensor is given by __repr__."""
        tensor = jet.Tensor(shape=[1, 2], indices=["i", "j"], data=[1j, 2 + 3j], dtype=dtype)
        assert tensor.__repr__() == "Size = 2\nIndices = {i  j}\nData = {(0,1)  (2,3)}"

    def test_fill_random(self, dtype):
        """Tests that a tensor is filled with random complex values."""
        tensor = jet.Tensor(shape=[2, 3], dtype=dtype)
        tensor.fill_random(seed=1)
        assert any(datum != tensor.data[0] for datum in tensor.data)
        assert all(-1 <= datum.real <= 1 for datum in tensor.data)
        assert all(-1 <= datum.imag <= 1 for datum in tensor.data)

    def test_init_indices_and_shape(self, dtype):
        """Tests that the indices and shape of a tensor can be modified."""
        tensor = jet.Tensor(shape=[2, 3], indices=["i", "j"], dtype=dtype)
        tensor.init_indices_and_shape(shape=[6], indices=["k"])
        assert tensor.indices == ["k"]
        assert tensor.shape == [6]
        assert tensor.index_to_dimension_map == {"k": 6}

    def test_get_value(self, dtype):
        """Tests that the value of a tensor at a multi-dimensional index can be retrieved."""
        tensor = jet.Tensor(shape=[2, 2], indices=["i", "j"], data=range(4), dtype=dtype)
        assert tensor.get_value(indices=[0, 0]) == 0
        assert tensor.get_value(indices=[0, 1]) == 1
        assert tensor.get_value(indices=[1, 0]) == 2
        assert tensor.get_value(indices=[1, 1]) == 3

    def test_set_value(self, dtype):
        """Tests that the value of a tensor at a multi-dimensional index can be modified."""
        tensor = jet.Tensor(shape=[2, 2], indices=["i", "j"], data=range(4), dtype=dtype)
        tensor.set_value(indices=[1, 1], value=9)
        assert tensor.get_value([1, 1]) == 9

    def test_is_scalar(self, dtype):
        """Tests that a scalar tensor can be detected."""
        scalar = jet.Tensor(dtype=dtype)
        vector = jet.Tensor(shape=[3], dtype=dtype)
        assert scalar.is_scalar() is True
        assert vector.is_scalar() is False

    def test_scalar(self, dtype):
        """Tests that the scalar value of a tensor can be retrieved."""
        tensor = jet.Tensor(shape=[2, 3], indices=["i", "j"], data=[9] + [0] * 5, dtype=dtype)
        assert tensor.scalar == 9

    def test_rename_index(self, dtype):
        """Tests that the index of a tensor can be renamed."""
        tensor = jet.Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], dtype=dtype)
        tensor.rename_index(2, "l")
        assert tensor.indices == ["i", "j", "l"]

    def test_add_tensor(self, dtype):
        """Tests that a pair of tensors can be added using class method."""
        tensor_1 = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(6))
        tensor_2 = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(0, 12, 2))
        have_tensor = tensor_1.add_tensor(tensor_2)
        want_tensor = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(0, 18, 3))
        assert have_tensor == want_tensor

    def test_conj(self, dtype):
        """Tests that the conjugate of a tensor can be taken using class method."""
        tensor = jet.Tensor(shape=[1, 2], indices=["i", "j"], data=[1, 2 + 3j])
        have_tensor = jet.conj(tensor)
        want_tensor = jet.Tensor(shape=[1, 2], indices=["i", "j"], data=[1, 2 - 3j])
        assert have_tensor == want_tensor

    def test_contract_with_tensor(self, dtype):
        """Tests that given tensor object can be contracted with another tensor."""
        tensor_1 = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
        tensor_2 = jet.Tensor(shape=[3, 4, 1], indices=["j", "k", "l"])
        have_tensor = tensor_1.contract_with_tensor(tensor_2)
        want_tensor = jet.Tensor(shape=[2, 1], indices=["i", "l"])
        assert have_tensor == want_tensor

    def test_slice_index(self, dtype):
        """Tests that a tensor can be sliced."""
        tensor = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
        have_tensor = tensor.slice_index("k", 3)
        want_tensor = jet.Tensor(shape=[2, 3], indices=["i", "j"])
        assert have_tensor == want_tensor

    def test_reshape(self, dtype):
        """Tests that a tensor can be reshaped."""
        tensor = jet.Tensor(shape=[3, 4], indices=["i", "j"])
        have_tensor = tensor.reshape([2, 6])
        want_tensor = jet.Tensor(shape=[2, 6])
        assert have_tensor == want_tensor


def test_add_tensors():
    """Tests that a pair of tensors can be added using module function."""
    tensor_1 = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(6))
    tensor_2 = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(0, 12, 2))
    have_tensor = jet.add_tensors(tensor_1, tensor_2)
    want_tensor = jet.Tensor(shape=[3, 2], indices=["i", "j"], data=range(0, 18, 3))
    assert have_tensor == want_tensor


def test_conj():
    """Tests that the conjugate of a tensor can be taken."""
    tensor = jet.Tensor(shape=[1, 2], indices=["i", "j"], data=[1, 2 + 3j])
    have_tensor_member = tensor.conj()
    have_tensor_module = jet.conj(tensor)
    want_tensor = jet.Tensor(shape=[1, 2], indices=["i", "j"], data=[1, 2 - 3j])
    assert have_tensor_member == want_tensor
    assert have_tensor_module == want_tensor


def test_contract_tensors():
    """Tests that a pair of tensors can be contracted."""
    tensor_1 = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
    tensor_2 = jet.Tensor(shape=[3, 4, 1], indices=["j", "k", "l"])
    have_tensor = jet.contract_tensors(tensor_1, tensor_2)
    want_tensor = jet.Tensor(shape=[2, 1], indices=["i", "l"])
    assert have_tensor == want_tensor


def test_slice_index():
    """Tests that a tensor can be sliced."""
    tensor = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
    have_tensor = jet.slice_index(tensor, "k", 3)
    want_tensor = jet.Tensor(shape=[2, 3], indices=["i", "j"])
    assert have_tensor == want_tensor


def test_reshape():
    """Tests that a tensor can be reshaped."""
    tensor = jet.Tensor(shape=[3, 4], indices=["i", "j"])
    have_tensor = jet.reshape(tensor, [2, 6])
    want_tensor = jet.Tensor(shape=[2, 6])
    assert have_tensor == want_tensor


class TestTranspose:
    def test_transpose_by_index(self):
        """Tests that a tensor can be transposed by index."""
        tensor = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
        have_tensor = tensor.transpose(["j", "k", "i"])
        want_tensor = jet.Tensor(shape=[3, 4, 2], indices=["j", "k", "i"])
        assert have_tensor == want_tensor

    def test_transpose_by_order(self):
        """Tests that a tensor can be transposed by order."""
        tensor = jet.Tensor(shape=[2, 3, 4], indices=["i", "j", "k"])
        have_tensor = tensor.transpose([1, 2, 0])
        want_tensor = jet.Tensor(shape=[3, 4, 2], indices=["j", "k", "i"])
        assert have_tensor == want_tensor
