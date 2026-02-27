

.PHONY: help install dev test clean migrate format lint

help: ## 显示帮助信息
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	@echo "📦 同步依赖环境..."
	uv sync	

dev: ## 启动 dev 模式
	@echo "🚀 启动开发服务器, dev模式..."
	bash run.dev.spzn_ai_structured_inference.sh

prod: ## 启动 prod 模式
	@echo "🚀 启动开发服务器, prod 模式..."
	bash run.dev.spzn_ai_structured_inference.sh

test: ## 运行测试
	@echo "🧪 运行测试..."
	uv run pytest tests/ -v


format: ## 代码检查
	@echo "🔍 运行代码检查..."
	uv run ruff format  src/


lint: ## 代码检查
	@echo "🔍 运行代码检查..."
	uv run ruff check src/


commit: ## 提交变更(取代git 流程)
	@echo "🔍 执行代码提交.."
	uv run cz commit

bump: ## 版本升级
	@echo "🔍 执行版本升级..."
	uv run cz bump