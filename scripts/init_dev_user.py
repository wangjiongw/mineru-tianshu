#!/usr/bin/env python3
"""
本地开发环境 - 初始化测试用户
快速创建固定的测试账户，用于本地开发
"""

import sys
import os
import fire
from pathlib import Path

# 添加 backend 到路径
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from auth.auth_db import AuthDB
from auth.models import UserCreate, UserRole


def init_test_user(db_path=None):
    """初始化测试用户"""

    if db_path is None:
        # 确保数据目录存在
        data_dir = Path(__file__).parent.parent / "data" / "db"
        data_dir.mkdir(parents=True, exist_ok=True)

        db_path = data_dir / "mineru_tianshu.db"
    if os.path.isfile(db_path):
        print(f"📊 使用数据库: {db_path}")
    else:
        print(f"❌数据库文件不存在: {db_path}")

    # 初始化认证数据库
    auth_db = AuthDB(str(db_path))

    # 检查是否已存在测试用户
    existing_user = auth_db.get_user_by_username("admin")
    if existing_user:
        print(f"⚠️  测试用户已存在: {existing_user.username}")
        print(f"   用户ID: {existing_user.user_id}")
        print(f"   邮箱: {existing_user.email}")
        print(f"   角色: {existing_user.role}")
        return

    # 创建测试管理员用户
    test_user = UserCreate(
        username="admin",
        email="admin@test.local",
        password="admin123",  # 测试用简单密码
        full_name="测试管理员",
        role=UserRole.ADMIN,
    )

    try:
        user = auth_db.create_user(test_user)
        print("✅ 测试用户创建成功！")
        print(f"   用户名: {user.username}")
        print(f"   邮箱: {user.email}")
        print(f"   密码: admin123")
        print(f"   角色: {user.role}")
        print(f"   用户ID: {user.user_id}")
        print()
        print("🔑 登录信息:")
        print("   URL: http://localhost:8000/api/v1/auth/login")
        print("   用户名: admin")
        print("   密码: admin123")
        print()
        print("📖 API 文档: http://localhost:8000/docs")
    except ValueError as e:
        print(f"❌ 创建用户失败: {e}")


if __name__ == "__main__":
    fire.Fire(init_test_user)
