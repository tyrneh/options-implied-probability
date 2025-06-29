# Migration Guide: Dependency Updates

## For Existing Users

If you're upgrading from a previous version of OIPD with the old dependency structure, here's what you need to know:

### What Changed?

1. **Reduced core dependencies** from 66+ to just 6 essential packages
2. **Streamlit is now optional** - only installed when you need the dashboard
3. **Version ranges** instead of pinned versions for better compatibility
4. **Removed unused packages** like `polygon-api-client`, `openpyxl`, `et-xmlfile`

### Migration Steps

#### 1. Clean Installation (Recommended)
```bash
# Uninstall old version and dependencies
pip uninstall oipd -y
pip freeze | grep -E "(altair|attrs|blinker|cachetools|certifi|charset-normalizer|click|commonmark|contourpy|cycler|decorator|entrypoints|et-xmlfile|exceptiongroup|fonttools|gitdb|GitPython|idna|importlib-metadata|iniconfig|Jinja2|jsonschema|kiwisolver|MarkupSafe|matplotlib|numpy|openpyxl|packaging|pandas|Pillow|pluggy|polygon-api-client|protobuf|pyarrow|pydeck|Pygments|Pympler|pyparsing|pyrsistent|pytest|python-dateutil|pytz|pytz-deprecation-shim|requests|rich|scipy|semver|six|smmap|streamlit|toml|tomli|toolz|tornado|typing_extensions|tzdata|tzlocal|urllib3|validators|websockets|zipp|traitlets)" | xargs pip uninstall -y

# Install new version
pip install oipd  # Core only
# OR
pip install oipd[dashboard]  # With Streamlit
# OR
pip install oipd[all]  # Everything
```

#### 2. Update Existing Installation
```bash
# Update to latest version
pip install --upgrade oipd

# If you need the dashboard
pip install oipd[dashboard]
```

### Code Changes Required

None! The API remains the same:

```python
import oipd.generate_pdf as op

# This still works exactly the same
result = op.run(
    input_data="your_data.csv",
    current_price=100.0,
    days_forward=30,
    risk_free_rate=0.03
)
```

### Benefits You'll See

1. **Faster installation** - Fewer dependencies to download
2. **Smaller footprint** - Less disk space used
3. **Better compatibility** - Version ranges allow for updates
4. **Clearer structure** - You know exactly what's installed

### Troubleshooting

#### "ModuleNotFoundError: No module named 'streamlit'"
You need to install the dashboard extra:
```bash
pip install oipd[dashboard]
```

#### "ImportError: cannot import name 'cli'"
The CLI functionality has been removed. Use the Python API directly:
```python
import oipd.generate_pdf as op
# Instead of oipd.cli
```

#### Dependency Conflicts
If you encounter dependency conflicts:
```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
pip install oipd[dashboard]
```

### For Package Maintainers

If you maintain a package that depends on OIPD:
- Update your dependency to `oipd>=0.0.5`
- If you use the dashboard, specify `oipd[dashboard]>=0.0.5`
- Remove any transitive dependencies you were explicitly listing 