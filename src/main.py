import os

os.environ['OMP_NUM_THREADS'] = '2'
from hybrid import HybridModel
from hybrid import project_path
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.dataset as ds
from mindspore import Model
from mindspore.train.callback import LossMonitor
from mindquantum import *
from abc import ABC, abstractmethod

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def conv(bit_up=0, bit_down=1, prefix='0'):
    _circ = Circuit()
    _circ += U3('theta00', 'phi00', 'lam00', bit_up)
    _circ += U3('theta01', 'phi01', 'lam01', bit_down)
    _circ += X.on(bit_down, bit_up)
    _circ += RY('theta10').on(bit_up)
    _circ += RZ('theta11').on(bit_down)
    _circ += X.on(bit_up, bit_down)
    _circ += RY('theta20').on(bit_up)
    _circ += X.on(bit_down, bit_up)
    _circ = add_prefix(_circ, prefix)

    return _circ


def ansa():
    ansatz = Circuit()
    ansatz += conv(0, 1, '00')
    ansatz += conv(2, 3, '01')
    ansatz += conv(4, 5, '02')
    ansatz += conv(6, 7, '03')

    ansatz += conv(7, 0, '10')
    ansatz += conv(1, 2, '11')
    ansatz += conv(3, 4, '12')
    ansatz += conv(5, 6, '13')

    ansatz += conv(1, 3, '20')
    ansatz += conv(5, 7, '21')

    ansatz += conv(7, 1, '30')
    ansatz += conv(3, 5, '31')

    ansatz += conv(3, 7, '40')

    ansatz += U3('theta400', 'phi401', 'lam402', 3)
    ansatz += U3('theta410', 'phi411', 'lam412', 7)

    return ansatz


# '''
class Main(HybridModel):
    def __init__(self):
        super().__init__()
        self.dataset = self.build_dataset(self.origin_x, self.origin_y, 10)
        self.qnet = MQLayer(self.build_grad_ops())
        self.model = self.build_model()
        self.checkpoint_name = os.path.join(project_path, "model.ckpt")

    def build_dataset(self, x, y, batch=None):
        # alpha = X[:, :3] * X[:, 1:]           # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出3个参数（因为有4个特征值），并储存在alpha中
        # X = np.append(X, alpha, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中

        # x=x.reshape((x.shape[0], -1))
        # alpha0 = x[:, :15] * x[:, 1:]           # 每一个样本中，利用相邻两个特征值计算出一个参数，即每一个样本会多出3个参数（因为有4个特征值），并储存在alpha中
        # alpha1 = x[:, 15] * x[:, 0]
        # alpha1 = alpha1.reshape([5000,1])
        # x = np.append(x, alpha0, axis=1)       # 在axis=1的维度上，将alpha的数据值添加到X的特征值中
        # x = np.append(x, alpha1, axis=1)

        train = ds.NumpySlicesDataset(
            {
                "image": x.reshape((x.shape[0], -1)),
                "label": y.astype(np.int32)
            },
            shuffle=False)
        if batch is not None:
            train = train.batch(batch)
        return train

    def build_grad_ops(self):
        circ = Circuit()
        for i in range(8):
            circ += RX(f'rx{i}').on(i)
            circ += RY(f'ry{i}').on(i)
        encoder = circ.as_encoder()
        # circ += UN(X, [1, 3, 5, 7,9,11,13,15], [0, 2, 4, 6,8,10,12,14])
        # circ += UN(X, [2, 4, 6,8,10,12,14], [1, 3, 5,7,9,11,13])
        # circ += UN(X, [1, 3, 5, 7], [0, 2, 4, 6])
        # circ += UN(X, [2, 4, 6], [1, 3, 5])
        # encoder = add_prefix(circ, 'e1') + add_prefix(circ, 'e2')

        ansatz = ansa().as_ansatz()
        total_circ = encoder + ansatz
        ham = [Hamiltonian(QubitOperator('Z3')), Hamiltonian(QubitOperator('Z7'))]
        sim = Simulator('projectq', total_circ.n_qubits)
        grad_ops = sim.get_expectation_with_grad(
            ham,
            total_circ,

            # encoder_params_name=encoder.params_name,
            # ansatz_params_name=ansatz.params_name,
            parallel_worker=5)
        #
        return grad_ops

    def build_model(self):
        self.loss = ms.nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.opti = ms.nn.Adam(self.qnet.trainable_params())
        self.model = Model(self.qnet, self.loss, self.opti)
        return self.model

    def train(self):
        self.model.train(1, self.dataset, callbacks=LossMonitor(per_print_times=15))

    def export_trained_parameters(self):
        qnet_weight = self.qnet.weight.asnumpy()
        ms.save_checkpoint(self.qnet, self.checkpoint_name)

    def load_trained_parameters(self):
        ms.load_param_into_net(self.qnet,
                               ms.load_checkpoint(self.checkpoint_name))

    def predict(self, origin_test_x) -> float:
        test_x = origin_test_x.reshape((origin_test_x.shape[0], -1))
        predict = self.model.predict(ms.Tensor(test_x))
        predict = predict.asnumpy().flatten() > 0
        return predict


# '''

if __name__ == '__main__':
    Main().train()
    print('hello')

