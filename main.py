# -*- coding: utf-8 -*-
"""
联邦学习主训练程序
"""

import os
import sys
import logging
import time
import json
import torch
import numpy as np
from datetime import datetime

# 导入自定义模块
from utils.config import get_args, print_args
from dataset.data_loader import get_federated_data
from models.model import create_model, print_model_info
from utils.federated import FederatedServer
from utils.utils import setup_logging, save_results, plot_training_curves


def setup_environment(args):
    """设置运行环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 设置设备
    device = torch.device(args.device)
    args.device = device

    # 设置日志
    setup_logging(args)

    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"fed_traffic_{args.aggregation}_{args.num_clients}clients_{timestamp}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir

    # 保存配置
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    logging.info(f"实验目录: {exp_dir}")
    logging.info(f"使用设备: {device}")

    return args


def main():
    """主函数"""
    # 解析参数
    args = get_args()

    # 设置环境
    args = setup_environment(args)

    # 打印配置
    print_args(args)

    try:
        # === 1. 数据加载 ===
        logging.info("=" * 60)
        logging.info("步骤1: 加载联邦数据")
        logging.info("=" * 60)

        federated_data, data_loader = get_federated_data(args)

        logging.info(f"成功加载 {federated_data['metadata']['num_clients']} 个客户端的数据")

        # 打印数据统计
        total_train_samples = sum([
            client_data['data_stats']['train_samples']
            for client_data in federated_data['clients'].values()
        ])
        total_test_samples = sum([
            client_data['data_stats']['test_samples']
            for client_data in federated_data['clients'].values()
        ])

        logging.info(f"训练样本总数: {total_train_samples}")
        logging.info(f"测试样本总数: {total_test_samples}")

        # === 2. 模型创建 ===
        logging.info("=" * 60)
        logging.info("步骤2: 创建全局模型")
        logging.info("=" * 60)

        # 创建全局模型（这里使用Transformer作为示例）
        global_model = create_model(args, model_type='transformer')
        global_model.to(args.device)

        # 打印模型信息
        print_model_info(global_model, args)

        # === 3. 联邦学习服务器初始化 ===
        logging.info("=" * 60)
        logging.info("步骤3: 初始化联邦学习服务器")
        logging.info("=" * 60)

        fed_server = FederatedServer(
            global_model=global_model,
            federated_data=federated_data,
            args=args
        )

        # === 4. 联邦训练 ===
        logging.info("=" * 60)
        logging.info("步骤4: 开始联邦训练")
        logging.info("=" * 60)

        best_loss = float('inf')
        patience_counter = 0
        training_history = []

        start_time = time.time()

        for round_num in range(args.rounds):
            round_start_time = time.time()

            # 执行一轮训练
            round_stats = fed_server.train_round(round_num)
            training_history.append(round_stats)

            round_time = time.time() - round_start_time

            # 定期评估
            if (round_num + 1) % args.eval_every == 0 or round_num == args.rounds - 1:
                eval_metrics = fed_server.evaluate_global_model()
                round_stats['eval_metrics'] = eval_metrics

                current_loss = eval_metrics['avg_test_loss']

                # 早停检查
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0

                    # 保存最佳模型
                    if args.save_model:
                        best_model_path = os.path.join(args.exp_dir, 'best_model.pth')
                        torch.save(global_model.state_dict(), best_model_path)
                        logging.info(f"保存最佳模型到: {best_model_path}")
                else:
                    patience_counter += 1

                # 打印详细评估结果
                logging.info(f"轮次 {round_num + 1} 评估结果:")
                logging.info(f"  测试损失: {eval_metrics['avg_test_loss']:.6f}")
                logging.info(f"  MSE: {eval_metrics['avg_mse']:.6f}")
                logging.info(f"  MAE: {eval_metrics['avg_mae']:.6f}")
                logging.info(f"  RMSE: {eval_metrics['avg_rmse']:.6f}")
                logging.info(f"  MAPE: {eval_metrics['avg_mape']:.2f}%")

                # 早停
                if patience_counter >= args.patience:
                    logging.info(f"早停触发！最佳损失: {best_loss:.6f}")
                    break

            # 打印轮次信息
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (round_num + 1)) * (args.rounds - round_num - 1)

            logging.info(f"轮次 {round_num + 1}/{args.rounds} 完成 "
                         f"(耗时: {round_time:.1f}s, 剩余: {eta / 60:.1f}min)")
            logging.info(f"  参与客户端: {round_stats['selected_clients']}")
            logging.info(f"  平均训练损失: {round_stats['avg_train_loss']:.6f}")
            if round_stats['avg_val_loss'] is not None:
                logging.info(f"  平均验证损失: {round_stats['avg_val_loss']:.6f}")

        total_time = time.time() - start_time

        # === 5. 最终评估 ===
        logging.info("=" * 60)
        logging.info("步骤5: 最终评估")
        logging.info("=" * 60)

        # 加载最佳模型进行最终评估
        if args.save_model and os.path.exists(os.path.join(args.exp_dir, 'best_model.pth')):
            best_model_state = torch.load(os.path.join(args.exp_dir, 'best_model.pth'))
            global_model.load_state_dict(best_model_state)
            logging.info("加载最佳模型进行最终评估")

        final_eval = fed_server.evaluate_global_model()

        # === 6. 保存结果 ===
        logging.info("=" * 60)
        logging.info("步骤6: 保存结果")
        logging.info("=" * 60)

        # 保存训练历史
        results = {
            'args': vars(args),
            'training_history': training_history,
            'eval_history': fed_server.eval_history,
            'final_evaluation': final_eval,
            'training_time': total_time,
            'best_loss': best_loss
        }

        save_results(results, args.exp_dir)

        # 绘制训练曲线
        try:
            plot_training_curves(training_history, args.exp_dir)
        except Exception as e:
            logging.warning(f"绘制训练曲线失败: {e}")

        # === 7. 打印最终结果 ===
        logging.info("=" * 60)
        logging.info("训练完成！")
        logging.info("=" * 60)

        logging.info(f"总训练时间: {total_time / 60:.1f} 分钟")
        logging.info(f"最佳测试损失: {best_loss:.6f}")

        logging.info("最终评估结果:")
        logging.info(f"  测试损失: {final_eval['avg_test_loss']:.6f}")
        logging.info(f"  MSE: {final_eval['avg_mse']:.6f}")
        logging.info(f"  MAE: {final_eval['avg_mae']:.6f}")
        logging.info(f"  RMSE: {final_eval['avg_rmse']:.6f}")
        logging.info(f"  MAPE: {final_eval['avg_mape']:.2f}%")

        # 打印每个客户端的结果
        logging.info("\n各客户端评估结果:")
        for client_result in final_eval['client_results']:
            client_id = client_result['client_id']
            mse = client_result['mse']
            mae = client_result['mae']
            logging.info(f"  客户端 {client_id}: MSE={mse:.6f}, MAE={mae:.6f}")

        logging.info(f"结果已保存到: {args.exp_dir}")

    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        raise
    finally:
        logging.info("程序结束")


if __name__ == "__main__":
    main()