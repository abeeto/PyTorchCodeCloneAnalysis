from unittest import TestCase

import torchtools
from torchtools.tensors.function import randbool_like
from torchtools.tensors.tensor_group import TensorGroup
import torch


class TensorFunctionTestCase(TestCase):
    def test_unsqueeze(self):
        tensor = torch.rand(2,3,4,5,6)
        tensor_1 = torchtools.tensors.function.unsqueeze(tensor, 3, 4)
        self.assertEqual(tensor_1.shape, (2,3,4,1,1,1,1,5,6))
        tensor_2 = torchtools.tensors.function.unsqueeze(tensor, 0, 4)
        self.assertEqual(tensor_2.shape, (1,1,1,1,2,3,4,5,6))
        tensor_3 = torchtools.tensors.function.unsqueeze(tensor, 5, 4)
        self.assertEqual(tensor_3.shape, (2,3,4,5,6,1,1,1,1))

class TensorGroupTestCase(TestCase):
    @staticmethod
    def prepare_data(dim: int=1) -> TensorGroup:
        record_num = 10
        feature_shape = (record_num,) + (5,) * (dim - 1)
        return TensorGroup(
            {
                'frame_id': torch.randint(200, size=(record_num,)),
                'features': torch.rand(size=feature_shape)
            }
        )

    def test_index(self):
        index = 3
        table = self.prepare_data()
        table_3 = table[index]
        self.assertTrue(torch.equal(table_3['frame_id'], table['frame_id'][index]))
        self.assertTrue(torch.equal(table_3['features'], table['features'][index]))

    def test_slice(self):
        slice = torch.tensor([2,3,4], dtype=torch.long)
        table = self.prepare_data()
        table_sliced = table[slice]
        self.assertTrue(torch.equal(table_sliced['frame_id'], table['frame_id'][slice]))
        self.assertTrue(torch.equal(table_sliced['features'], table['features'][slice]))

    def test_ge(self):
        value = 0.5
        table = self.prepare_data()
        table_ge = table.ge_select('features', value)
        self.assertTrue(
            torch.equal(
                table_ge['features'],
                table['features'].masked_select(table['features'].ge(value))
            )
        )
        self.assertTrue(
            torch.equal(
                table_ge['frame_id'],
                table['frame_id'].masked_select(table['features'].ge(value))
            )
        )

    def test_mask_select(self):
        table = self.prepare_data(dim=2)
        mask = randbool_like(table['frame_id'])
        table_masked = table.masked_select(mask)
        self.assertTrue(
            torch.equal(
                table_masked['frame_id'],
                table['frame_id'].masked_select(mask)
            )
        )
        self.assertTrue(
            torch.equal(
                table_masked['features'],
                table['features'].masked_select(mask.unsqueeze(1))
            )
        )

    def test_index_select(self):
        table = self.prepare_data(dim=2)
        indices = torch.tensor([0, 3, 2]).long()
        table_selected = table.index_select(indices)
        self.assertTrue(
            torch.equal(
                table_selected['frame_id'],
                table['frame_id'].index_select(dim=0, idx=indices)
            )
        )
        self.assertTrue(
            torch.equal(
                table_selected['features'],
                table['features'].index_select(dim=0, idx=indices)
            )
        )