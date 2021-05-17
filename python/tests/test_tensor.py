import jet
import pytest


@pytest.mark.parametrize("Tensor", [jet.Tensor32, jet.Tensor64])
class TestTensor:
    def test_default_constructor(self, Tensor):
        """Tests that the default constructor is called."""
        tensor = Tensor()
        assert tensor.indices == []
        assert tensor.shape == []
        assert tensor.data == [0]
        assert tensor.index_to_dimension_map == {}

    def test_shape_constructor(self, Tensor):
        """Tests that the (shape) constructor is called."""
        tensor = Tensor(shape=[1, 2, 3])
        assert tensor.indices == ["?a", "?b", "?c"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 0, 0, 0, 0, 0]
        assert tensor.index_to_dimension_map == {"?a": 1, "?b": 2, "?c": 3}

    def test_shape_index_constructor(self, Tensor):
        """Tests that the (shape, indices) constructor is called."""
        tensor = Tensor(shape=[1, 2, 3], indices=["i", "j", "k"])
        assert tensor.indices == ["i", "j", "k"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 0, 0, 0, 0, 0]
        assert tensor.index_to_dimension_map == {"i": 1, "j": 2, "k": 3}

    def test_shape_index_data_constructor(self, Tensor):
        """Tests that the (shape, indices, data) constructor is called."""
        tensor = Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], data=range(6))
        assert tensor.indices == ["i", "j", "k"]
        assert tensor.shape == [1, 2, 3]
        assert tensor.data == [0, 1, 2, 3, 4, 5]
        assert tensor.index_to_dimension_map == {"i": 1, "j": 2, "k": 3}

    def test_copy_constructor(self, Tensor):
        """Tests that the copy constructor is called."""
        tensor_1 = Tensor(shape=[1, 2, 3], indices=["i", "j", "k"], data=range(6))
        tensor_2 = Tensor(other=tensor_1)
        assert tensor_1 == tensor_2

    def test_set_shape(self, Tensor):
        """Tests that the shape of a tensor can be modified."""
        tensor = Tensor(shape=[1, 2, 3, 4])
        tensor.shape = [2, 6, 1, 1, 1]
        assert tensor.shape == [2, 6, 1, 1, 1]

    def test_getitem(self, Tensor):
        """Tests that the data of a tensor can be randomly accessed using __getitem__."""
        tensor = Tensor(shape=[3], indices=["i"], data=range(3))
        assert tensor[0] == 0
        assert tensor[1] == 1
        assert tensor[2] == 2

    def test_len(self, Tensor):
        """Tests that the size of a tensor can be retrieved using __len__."""
        tensor = Tensor(shape=[2, 3, 4])
        assert len(tensor) == 2 * 3 * 4

    def test_repr(self, Tensor):
        """Tests that the string representation of a tensor is given by __repr__."""
        tensor = Tensor(shape=[1, 2], indices=["i", "j"], data=[1j, 2 + 3j])
        assert tensor.__repr__() == "Size=2\nIndices={i  j}\nData={(0,1)  (2,3)}\n"

    def test_fill_random(self, Tensor):
        """Tests that a tensor is filled with random complex values."""
        tensor = Tensor(shape=[2, 3])
        tensor.fill_random(seed=1)
        assert any(datum != tensor.data[0] for datum in tensor.data)
        assert all(-1 <= datum.real <= 1 for datum in tensor.data)
        assert all(-1 <= datum.imag <= 1 for datum in tensor.data)

    def test_init_indices_and_shape(self, Tensor):
        """Tests that the indices and shape of a tensor can be modified."""
        tensor = Tensor(shape=[2, 3], indices=["i", "j"])
        tensor.init_indices_and_shape(shape=[6], indices=["k"])
        assert tensor.indices == ["k"]
        assert tensor.shape == [6]
        assert tensor.index_to_dimension_map == {"k": 6}

    def test_get_value(self, Tensor):
        """Tests that the value of a tensor at a multi-dimensional index can be retrieved."""
        tensor = Tensor(shape=[2, 2], indices=["i", "j"], data=range(4))
        assert tensor.get_value(indices=[0, 0]) == 0
        assert tensor.get_value(indices=[1, 0]) == 1
        assert tensor.get_value(indices=[0, 1]) == 2
        assert tensor.get_value(indices=[1, 1]) == 3

    def test_set_value(self, Tensor):
        """Tests that the value of a tensor at a multi-dimensional index can be modified."""
        tensor = Tensor(shape=[2, 2], indices=["i", "j"], data=range(4))
        tensor.set_value(indices=[1, 1], value=9)
        assert tensor.get_value([1, 1]) == 9

    def test_is_scalar(self, Tensor):
        """Tests that a scalar tensor can be detected."""
        scalar = Tensor()
        vector = Tensor(shape=[3])
        assert scalar.is_scalar() is True
        assert vector.is_scalar() is False

    def test_scalar(self, Tensor):
        """Tests that the scalar value of a tensor can be retrieved."""
        tensor = Tensor(shape=[2, 3], indices=["i", "j"], data=[9] + [0] * 5)
        assert tensor.scalar == 9

    def test_rename_index(self, Tensor):
        """Tests that the index of a tensor can be renamed."""
        tensor = Tensor(shape=[1, 2, 3], indices=["i", "j", "k"])
        tensor.rename_index(2, "l")
        assert tensor.indices == ["i", "j", "l"]


def test_conj():
    """Tests that the conjugate of a tensor can be taken."""
    tensor = jet.Tensor64(shape=[1, 2], indices=["i", "j"], data=[1, 2 + 3j])
    have_tensor = jet.conj(tensor)
    want_tensor = jet.Tensor64(shape=[1, 2], indices=["i", "j"], data=[1, 2 - 3j])
    assert have_tensor == want_tensor


def test_contract_tensors():
    """Tests that a pair of tensors can be contracted."""
    tensor_1 = jet.Tensor64(shape=[2, 3, 4], indices=["i", "j", "k"])
    tensor_2 = jet.Tensor64(shape=[3, 4, 1], indices=["j", "k", "l"])
    have_tensor = jet.contract_tensors(tensor_1, tensor_2)
    want_tensor = jet.Tensor64(shape=[2, 1], indices=["i", "l"])
    assert have_tensor == want_tensor


def test_slice_index():
    """Tests that a tensor can be sliced."""
    tensor = jet.Tensor64(shape=[2, 3, 4], indices=["i", "j", "k"])
    have_tensor = jet.slice_index(tensor, "k", 3)
    want_tensor = jet.Tensor64(shape=[2, 3], indices=["i", "j"])
    assert have_tensor == want_tensor


def test_reshape():
    """Tests that a tensor can be reshaped."""
    tensor = jet.Tensor64(shape=[3, 4], indices=["i", "j"])
    have_tensor = jet.reshape(tensor, [2, 6])
    want_tensor = jet.Tensor64(shape=[2, 6])
    assert have_tensor == want_tensor


class TestTranspose:
    def test_transpose_by_index(self):
        """Tests that a tensor can be transposed by index."""
        tensor = jet.Tensor64(shape=[2, 3, 4], indices=["i", "j", "k"])
        have_tensor = jet.transpose(tensor, ["j", "k", "i"])
        want_tensor = jet.Tensor64(shape=[3, 4, 2], indices=["j", "k", "i"])
        assert have_tensor == want_tensor

    def test_transpose_by_order(self):
        """Tests that a tensor can be transposed by order."""
        tensor = jet.Tensor64(shape=[2, 3, 4], indices=["i", "j", "k"])
        have_tensor = jet.transpose(tensor, [1, 2, 0])
        want_tensor = jet.Tensor64(shape=[3, 4, 2], indices=["j", "k", "i"])
        assert have_tensor == want_tensor
