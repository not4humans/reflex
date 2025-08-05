# 🎯 Agentic Skill Compiler

**Human-inspired skill compilation loop for multi-agent systems**

This system automatically identifies, validates, and compiles reusable skills from agent execution traces, following the eight-phase compilation loop described in our research paper.

## 🚀 Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Generate sample data
asc demo --tasks 10

# View traces
asc traces

# Mine skills
asc mine --limit 100
```

## 📊 System Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ Complete | Declarative scaffolding |
| 2 | ✅ Complete | Trace collection and storage |
| 3 | ✅ Complete | Pattern mining and skill identification |
| 4 | ⏳ Ready | Skill compilation/distillation |
| 5-8 | 📋 Planned | Validation, retrieval, perturbation, registry |

## 🏗️ Architecture

```
asc/
├── core/           # Agent execution and data models
├── storage/        # SQLite-based trace storage with WAL mode
├── mining/         # PrefixSpan pattern mining and filtering
├── tools/          # Tool registry with realistic cost modeling
└── config.py       # Centralized configuration management
```

## 🔧 Configuration

The system is designed for real-world deployment with configurable parameters:

**Development Environment:**
```python
config.mining.min_support_percent = 0.5   # Find more patterns
config.mining.min_success_rate = 0.70     # Accept some failures
config.mining.max_cost_ratio = 1.2        # Don't require savings
```

**Production Environment:**
```python
config.mining.min_support_percent = 3.0   # Only reliable patterns
config.mining.min_success_rate = 0.90     # High reliability
config.mining.max_cost_ratio = 0.70       # Require 30% savings
```

## 📈 Examples

See `examples/` directory for:
- Configuration tuning guidance
- Performance analysis
- Integration patterns

## 🧪 Testing

```bash
# Comprehensive system test
python tests/test_comprehensive.py

# Configuration tuning demo
python examples/config_tuning.py
```

## 🎯 Research Validation

This implementation validates key research hypotheses:

- ✅ **<0.1% trace loss** achieved with SQLite WAL mode
- ✅ **Realistic cost modeling** influences skill selection
- ✅ **Configurable quality thresholds** prevent noise optimization
- ✅ **Production-ready architecture** supports real-world deployment

## 📚 Documentation

- `SETUP.md` - Setup and configuration guide
- `COMPREHENSIVE_TEST_RESULTS.md` - Detailed test analysis
- `config/default.json` - Configuration reference

## 🛠️ Next Steps

Ready for **Phase 4: Skill Compilation** - converting identified patterns into:
- Python macros for immediate execution
- LoRA adapters for model fine-tuning
- Custom tool compositions

## 📄 License

MIT License
