@echo off
title Claude Code + DeepSeek - Seletor de Modelo
color 0A

:: Tenta definir codificação UTF-8 para exibir caracteres corretamente (opcional)
chcp 65001 >nul 2>nul

:: ============================================
:: 1. DEFINIÇÃO DA CHAVE API (SUBSTITUA PELA SUA)
:: ============================================
set DEEPSEEK_API_KEY=sk-fa04bf20988d4c168e736c9f98ecd48f

:: ============================================
:: 2. CONFIGURAÇÕES BASE (compartilhadas)
:: ============================================
set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
set ANTHROPIC_AUTH_TOKEN=%DEEPSEEK_API_KEY%
set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

:: ============================================
:: 3. MENU INTERATIVO
:: ============================================
:menu
cls
echo ==================================================
echo    Claude Code + DeepSeek - Escolha o modelo
echo ==================================================
echo.
echo  [1] DeepSeek V4 FLASH   (rapido, baixo custo)
echo  [2] DeepSeek V4 PRO     (completo, raciocinio avancado)
echo  [3] Sair
echo.
set /p escolha="Digite 1, 2 ou 3: "

if "%escolha%"=="1" goto config_flash
if "%escolha%"=="2" goto config_pro
if "%escolha%"=="3" exit /b
echo Opcao invalida! Tente novamente.
timeout /t 2 >nul
goto menu

:: ============================================
:: 4. PERFIL FLASH (rápido e econômico)
:: ============================================
:config_flash
set ANTHROPIC_MODEL=deepseek-chat
set ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat
set API_TIMEOUT_MS=300000
set ANTHROPIC_MAX_TOKENS=4096
set ANTHROPIC_TEMPERATURE=0.8
set ANTHROPIC_TOP_P=0.95
set ANTHROPIC_THINKING_TYPE=disabled
echo.
echo [PERFIL FLASH] Configurado - modelo: deepseek-chat
echo - Timeout: 5 minutos ^| Max tokens: 4096 ^| Temperatura: 0.8
echo.
goto iniciar

:: ============================================
:: 5. PERFIL PRO (completo, para tarefas complexas)
:: ============================================
:config_pro
set ANTHROPIC_MODEL=deepseek-reasoner
set ANTHROPIC_SMALL_FAST_MODEL=deepseek-chat
set API_TIMEOUT_MS=900000
set ANTHROPIC_MAX_TOKENS=32768
set ANTHROPIC_TEMPERATURE=0.3
set ANTHROPIC_TOP_P=0.9
set ANTHROPIC_THINKING_TYPE=enabled
set ANTHROPIC_BUDGET_TOKENS=16000
echo.
echo [PERFIL PRO] Configurado - modelo: deepseek-reasoner
echo - Timeout: 15 minutos ^| Max tokens: 32768 ^| Temperatura: 0.3
echo - Raciocinio profundo ativado (thinking)
echo.
goto iniciar

:: ============================================
:: 6. INICIALIZAÇÃO DO CLAUDE CODE
:: ============================================
:iniciar
echo Iniciando Claude Code com as configuracoes acima...
echo.
echo (Pressione Ctrl+C a qualquer momento para encerrar)
echo.
timeout /t 2 >nul
claude

:: Ao sair do Claude, volta ao menu (opcional)
echo.
echo ==================================================
echo   Claude Code encerrado.
echo ==================================================
echo.
echo Pressione qualquer tecla para voltar ao menu...
pause >nul
goto menu