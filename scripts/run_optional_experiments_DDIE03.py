#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDIE03 可选实验：NASA 电池 LOOCV、IGBT 两折、RUL LOOCV。
支持分步执行，避免长时间训练后因小错误重跑：
  --step 1|2|3  只跑某一步
  --train_only   步骤 1/2 仅训练各折并保存 checkpoint，不评估
  --skip_train   步骤 1/2 仅用已有 checkpoint 做评估与汇总（出错后可单独重跑此步）
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="DDIE03 可选实验：分步执行 LOOCV / IGBT 2-fold / RUL LOOCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
分步示例（避免长训练后因评估错误重跑）:
  步骤1 仅训练:  python run_optional_experiments_DDIE03.py --step 1 --train_only
  步骤1 仅评估:  python run_optional_experiments_DDIE03.py --step 1 --skip_train
  步骤2 仅训练:  python run_optional_experiments_DDIE03.py --step 2 --train_only
  步骤2 仅评估:  python run_optional_experiments_DDIE03.py --step 2 --skip_train
  步骤3:         python run_optional_experiments_DDIE03.py --step 3
  从某步继续:    python run_optional_experiments_DDIE03.py --step 2  (只跑步骤2)
        """,
    )
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="只运行指定步骤（1=LOOCV, 2=IGBT两折, 3=RUL LOOCV）")
    parser.add_argument("--loocv_only", action="store_true", help="仅运行步骤1（同 --step 1）")
    parser.add_argument("--igbt_only", action="store_true", help="仅运行步骤2（同 --step 2）")
    parser.add_argument("--rul_only", action="store_true", help="仅运行步骤3（同 --step 3）")
    parser.add_argument("--train_only", action="store_true", help="步骤1/2 仅训练各折并保存 checkpoint，不进行评估与汇总")
    parser.add_argument("--skip_train", action="store_true", help="步骤1/2 不训练，仅用已有 checkpoint 做评估与汇总")
    parser.add_argument("--max_epochs", type=int, default=200, help="每折最大训练轮数")
    args = parser.parse_args()

    # 确定要跑哪些步骤
    if args.step:
        steps_to_run = [args.step]
    elif args.loocv_only:
        steps_to_run = [1]
    elif args.igbt_only:
        steps_to_run = [2]
    elif args.rul_only:
        steps_to_run = [3]
    else:
        steps_to_run = [1, 2, 3]

    ran = []

    if 1 in steps_to_run:
        print("\n" + "=" * 50 + "\n[1/3] NASA 电池 LOOCV (4 折)\n" + "=" * 50)
        import run_loocv_nasa_battery_DDIE03 as m1
        argv = ["run_loocv_nasa_battery_DDIE03.py"]
        if args.train_only:
            argv.append("--train_only")
        if args.skip_train:
            argv.append("--skip_train")
        if args.max_epochs != 200:
            argv.extend(["--max_epochs", str(args.max_epochs)])
        old = sys.argv
        sys.argv = argv
        try:
            m1.main()
            ran.append("nasa_battery_LOOCV")
        finally:
            sys.argv = old

    if 2 in steps_to_run:
        print("\n" + "=" * 50 + "\n[2/3] NASA IGBT 两折交叉验证\n" + "=" * 50)
        import run_2fold_igbt_DDIE03 as m2
        argv = ["run_2fold_igbt_DDIE03.py"]
        if args.train_only:
            argv.append("--train_only")
        if args.skip_train:
            argv.append("--skip_train")
        if args.max_epochs != 200:
            argv.extend(["--max_epochs", str(args.max_epochs)])
        old = sys.argv
        sys.argv = argv
        try:
            m2.main()
            ran.append("nasa_igbt_2fold")
        finally:
            sys.argv = old

    if 3 in steps_to_run:
        print("\n" + "=" * 50 + "\n[3/3] RUL LOOCV (4 折)\n" + "=" * 50)
        import run_rul_loocv_DDIE03 as m3
        old = sys.argv
        sys.argv = ["run_rul_loocv_DDIE03.py"]
        try:
            m3.main()
            ran.append("rul_LOOCV")
        finally:
            sys.argv = old

    print("\n可选实验已完成:", ran)
    return 0


if __name__ == "__main__":
    sys.exit(main())
