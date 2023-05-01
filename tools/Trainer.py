import os
import time
import math
import wandb
import numpy as np

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):
    def __init__(self, model, train_loader, train_sampler, test_loader, test_sampler, optimizer, loss_fn, \
                 model_name, type='mv'):
        self.model = model
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.test_loader = test_loader
        self.test_sampler = test_sampler

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.model_name = model_name
        self.type = type

    def train(self, rank, logger, args):
        if rank == 0:
            os.environ["WANDB_BASE_URL"] = args.wb_url
            wandb.login(key=args.wb_key)
            wandb.init(project=args.proj_name, name=args.exp_name)

        logger.write('Start DDP training on %s ...' % args.dataset, rank=rank)

        test_best_class_epoch = 0
        test_best_inst_epoch = 0
        test_best_class_acc = .0
        test_best_inst_acc = .0

        if self.type == 'sv':
            epochs = args.base_model_epochs
        else:
            epochs = args.epochs

        for epoch in range(epochs):
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)

            if self.model_name == 'view_gcn':
                if epoch == 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                if epoch > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * ( 1 + math.cos(epoch * math.pi / 15))
            else:
                if epoch > 0 and (epoch + 1) % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5

            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            # train one epoch
            out_data = None
            in_data = None
            num_shapes = len(self.train_loader.dataset.filepaths) // args.num_views
            train_interval = AverageMeter()

            for i, data in enumerate(self.train_loader):
                # data: (class_id, imgs_within_a_batch, imgs_path_within_a_batch)

                if self.model_name == 'view-gcn' and epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (num_shapes // 20))
                if self.model_name == 'view-gcn':
                    B, V, C, H, W = data[1].size()
                    in_data = data[1].view(-1, C, H, W).to(rank)
                else:
                    in_data = data[1].to(rank)
                target = data[0].to(rank).long()
                # 4*(10+5) 是在干什么？最后用这个做分类损失用，40个物体类别，20个视角
                #   这一步应该就是 view-gcn 里 `L_view` 这一项loss
                target_ = target.unsqueeze(1).repeat(1, 4*(10+5)).view(-1)
                # target_ = target.unsqueeze(1).repeat(1, args.num_obj_classes + args.num_views).view(-1)
                self.optimizer.zero_grad()

                start = time.time()
                if self.model_name == 'view-gcn':
                    out_data, F_score,F_score2= self.model(in_data)
                    # out_data_ 的类别数 和 target_ 对不上啊
                    out_data_ = torch.cat((F_score, F_score2), 1).view(-1, args.num_obj_classes)
                    train_loss = self.loss_fn(out_data, target)+ self.loss_fn(out_data_, target_)
                else:
                    out_data = self.model(in_data)
                    train_loss = self.loss_fn(out_data, target)

                train_loss.backward()
                B = out_data.shape[0]
                train_interval.update(time.time() - start, n=B)
                self.optimizer.step()

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                train_acc = correct_points.float() / results.size()[0]
                #print('lr = ', str(param_group['lr']))

                if i % args.print_freq == 0:
                    logger.write(f'Epoch: {epoch}/{epochs}, Batch: {i}/{len(self.train_loader)}, '
                                f'Loss : {train_loss.item()}, Accuracy: {train_acc} ', rank=rank)
            
            # --- Test
            logger.write('Start testing on %s ...' % args.dataset, rank=rank)
            # 为什么这里出不来结果，是哪里出了问题，只可能是函数内部出了问题
            test_loss, test_overall_acc, test_mean_class_acc, test_interval = self.update_validation_accuracy(rank, args)

            logger.write('Got test instance accuracy on [%s]: %f' % (args.dataset, test_overall_acc), rank=rank)
            logger.write('Got test class accuracy on [%s]: %f' % (args.dataset, test_mean_class_acc), rank=rank)
                
            if rank == 0:
                if test_overall_acc >= test_best_inst_acc:
                    test_best_inst_acc = test_overall_acc
                    test_best_inst_epoch = epoch
                    logger.write(f'Find new best Instance Accuracy: <{test_best_inst_acc}> at epoch [{test_best_inst_epoch}] !', rank=rank)
                    logger.write('Saving best model ...', rank=rank)
                    
                    save_dict = self.model.module.state_dict()
                    save_path = os.path.join('runs', args.task, args.proj_name, args.exp_name, 'weights', f'{self.type}_model_best.pth')
                    torch.save(save_dict, save_path)

                if test_mean_class_acc > test_best_class_acc:
                    test_best_class_acc = test_mean_class_acc
                    test_best_class_epoch = epoch

                wandb_log = dict()
                wandb_log['lr'] = lr
                wandb_log['train_loss'] = train_loss.item()
                wandb_log['train_acc'] = train_acc
                wandb_log['train_interval'] = train_interval.avg
                wandb_log['test_loss'] = test_loss
                wandb_log['test_inst_acc'] = test_overall_acc
                wandb_log['test_best_inst_acc'] = test_best_inst_acc
                wandb_log['test_class_acc'] = test_mean_class_acc
                wandb_log['test_best_class_acc'] = test_best_class_acc
                wandb_log['test_best_inst_epoch'] = test_best_inst_epoch
                wandb_log['test_best_class_epoch'] = test_best_class_epoch
                wandb_log['test_interval'] = test_interval.avg
                wandb.log(wandb_log)

        if rank == 0:
            logger.write(f'Final best Instance Accuracy on [{args.dataset}]: <{test_best_inst_acc}> at epoch [{test_best_inst_epoch}] !', rank=rank)
            logger.write(f'Final best Class Accuracy on [{args.dataset}]: <{test_best_class_acc}> at epoch [{test_best_class_epoch}] !', rank=rank)
            logger.write(f'End of DDP training on [{args.dataset}] ...', rank=rank)
            wandb.finish()

    def update_validation_accuracy(self, rank, args):
        with torch.no_grad():
            all_correct_points = 0
            all_points = 0
            
            test_interval = AverageMeter()
            wrong_class = np.zeros(args.num_obj_classes)
            samples_class = np.zeros(args.num_obj_classes)
            all_loss = 0

            self.model.eval()

            for data in self.test_loader:

                if self.model_name == 'view-gcn':
                    B, V, C, H, W = data[1].size()
                    in_data = data[1].view(-1, C, H, W).to(rank)
                else:  # 'svcnn'
                    in_data = data[1].to(rank)
                target = data[0].to(rank)

                start = time.time()
                if self.model_name == 'view-gcn':
                    out_data, F1, F2 = self.model(in_data)
                else:
                    out_data = self.model(in_data)
                
                all_loss += self.loss_fn(out_data, target).cpu().numpy()
                B = out_data.shape[0]
                test_interval.update(time.time() - start, n=B)

                pred = torch.max(out_data, 1)[1]
                results = pred == target

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().numpy()):
                        wrong_class[target.cpu().numpy().astype('int')[i]] += 1
                    samples_class[target.cpu().numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]

            class_acc = (samples_class - wrong_class) / samples_class
            val_mean_class_acc = np.mean(class_acc)
            acc = all_correct_points.float() / all_points
            val_overall_acc = acc.cpu().numpy()
            loss = all_loss / len(self.test_loader)

            return loss, val_overall_acc, val_mean_class_acc, test_interval
