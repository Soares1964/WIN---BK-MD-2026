#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de diagnóstico para verificar instalação
"""

import sys
import platform

print("="*60)
print("🔍 DIAGNÓSTICO DO SISTEMA")
print("="*60)
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Processor: {platform.processor()}")
print()

# Verifica pacotes
packages = ['numpy', 'pandas', 'numba', 'PySide6']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'desconhecida')
        print(f"✅ {pkg}: {version}")
    except ImportError as e:
        print(f"❌ {pkg}: Não instalado - {e}")

print()

# Testa Numba
try:
    from numba import njit
    
    @njit
    def test_func(x):
        return x * 2
    
    result = test_func(21)
    print(f"✅ Numba JIT funcionando: 21*2 = {result}")
except Exception as e:
    print(f"❌ Numba JIT falhou: {e}")

print("="*60)