# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class YOLOXModeSwitchHook(Hook):
    """Switch the mode of YOLOX during training.

    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.

    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Defaults to 15.
       skip_type_keys (Sequence[str], optional): Sequence of type string to be
            skip pipeline. Defaults to ('Mosaic', 'RandomAffine', 'MixUp').
    """

    def __init__(
        self,
        num_last_epochs: int = 15,
        skip_type_keys: Sequence[str] = ('Mosaic', 'RandomAffine', 'MixUp')
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        epoch_to_be_switched = ((epoch + 1) >=
                                runner.max_epochs - self.num_last_epochs)
        if epoch_to_be_switched and not self._has_switched:
            runner.logger.info('No mosaic and mixup aug now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            runner.logger.info('Add additional L1 loss now!')
            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_l1 = True
            else:
                model.bbox_head.use_l1 = True
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True


@HOOKS.register_module()
class YOLOXModeXfeatSwitchHook(YOLOXModeSwitchHook):
    def __init__(
        self,
        num_last_epochs: int = 15,
        skip_type_keys: Sequence[str] = ('Mosaic', 'RandomAffine', 'MixUp'),
        num_starting_epochs=10
    ) -> None:
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False
        self._has_switched = False
        self.num_starting_epochs=num_starting_epochs
        self._has_switched_xfeat = False  ##是否恢复加入xfeat
        self.has_restored = False ##是否恢复加入马赛克等

    def before_train_epoch(self, runner) -> None:
        """Close mosaic and mixup augmentation and switches to use L1 loss."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        epoch_to_be_switched = ((epoch + 1) >=
                                runner.max_epochs - self.num_last_epochs)
        if_use_xfeat = (epoch <self.num_starting_epochs)
        if_use_mosaic = (epoch_to_be_switched or if_use_xfeat) is False

        '''
        xfeat 环节
        '''
        if if_use_xfeat and not self._has_switched_xfeat:
            ##用xfeat时候必然没有这些增强
            runner.logger.info('No mosaic and mixup aug now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            runner.logger.info('Add additional xfeat loss now!')
            if hasattr(model, 'detector'):
                model.detector.use_xfeat = True
            else:
                model.use_xfeat = True
            self._has_switched_xfeat=True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
        ''' mosaic 环节 '''
        if if_use_mosaic and not self.has_restored:
            ##用xfeat时候必然没有这些增强
            runner.logger.info('restore mosaic and mixup aug now!')
            train_loader.dataset._skip_type_keys=None
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self.has_restored=True
            if hasattr(model, 'detector'):
                model.detector.use_xfeat = False
            else:
                model.use_xfeat = False
            runner.logger.info('Shutdown xfeat loss now!')
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

        ''' 最后 环节 '''
        if epoch_to_be_switched and not self._has_switched:
            runner.logger.info('No mosaic and mixup aug now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            runner.logger.info('Add additional L1 loss now!')
            if hasattr(model, 'detector'):
                model.detector.bbox_head.use_l1 = True
            else:
                model.bbox_head.use_l1 = True
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True
