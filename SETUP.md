# Agentic Skill Compiler Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (run this every time you work on the project)
source venv/bin/activate

# Install package in development mode
pip install -e .

# Run demo to generate sample traces
asc demo --tasks 10

# Check traces
asc traces

# Mine patterns (Phase 3)
asc mine --limit 100
```

## System Status

✅ **Phase 1**: Declarative scaffolding - Complete  
✅ **Phase 2**: Trace collection and storage - Complete  
✅ **Phase 3**: Pattern mining and skill identification - Complete  
⏳ **Phase 4**: Skill compilation/distillation - Ready to implement  

## Architecture

```
asc/
├── core/           # Agent and data models
├── storage/        # SQLite-based trace storage
├── mining/         # Pattern mining and skill extraction
├── tools/          # Tool registry and cost modeling
└── config.py       # Configuration management

examples/           # Usage examples and demos
tests/              # Comprehensive test suite
config/             # Configuration templates
```

## Configuration

The system uses a layered configuration approach:

1. **Default**: `config/default.json` - Base configuration
2. **Environment**: `config.local.json` - Your local overrides
3. **Runtime**: Python code can modify config at runtime

Key parameters to tune:
- `mining.min_support_percent`: Pattern frequency threshold
- `mining.min_success_rate`: Quality threshold for skills
- `mining.max_cost_ratio`: Cost efficiency requirement

## Next Steps

Ready for **Phase 4: Skill Compilation** - converting identified patterns into executable skills.
