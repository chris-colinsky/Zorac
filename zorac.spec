# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['run_zorac.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('zorac/*.py', 'zorac'),
    ],
    hiddenimports=[
        'tiktoken_ext.openai_public',
        'tiktoken_ext',
        'rich',
        'openai',
        'dotenv',
        'zorac',
        'zorac.main',
        'zorac.config',
        'zorac.console',
        'zorac.llm',
        'zorac.session',
        'zorac.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='zorac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
