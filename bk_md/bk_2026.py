import os
import sys
import shutil
import zipfile
import re
import subprocess
from datetime import datetime
from pathlib import Path

# ==============================================================================
# ⚙️ CONFIGURAÇÕES E CONSTANTES GLOBAIS
# ==============================================================================

# Localização baseada no arquivo do script (mais seguro que CWD)
SCRIPT_ATUAL = Path(__file__).resolve()
PASTA_RAIZ = SCRIPT_ATUAL.parent
NOME_PROJETO = PASTA_RAIZ.name.lower().replace(" ", "_").replace("-", "_")
PASTA_BACKUP = PASTA_RAIZ / "backup"

# Cria a pasta de backup se não existir
PASTA_BACKUP.mkdir(exist_ok=True)

# 🚫 LISTA NEGRA: Pastas que JAMAIS entrarão no backup ou leitura
# Isso garante que venv, node_modules e git sejam ignorados na raiz
PASTAS_IGNORAR = {
    '.venv', 'venv', 'env', 'virtualenv',  # Ambientes Virtuais
    '__pycache__', '.git', '.idea', '.vscode', # Configs IDE/Git
    'node_modules', 'dist', 'build', 'site-packages', # Build/Deps
    'backup' # Ignora a própria pasta de backup
}

# Extensões irrelevantes
EXTENSOES_IGNORAR = {'.pyc', '.pyo', '.pyd', '.tmp', '.log', '.zip', '.exe', '.ds_store'}

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

# ==============================================================================
# 1. 📦 BACKUP OTIMIZADO (SEM VENV)
# ==============================================================================
def backup_projeto(silencioso=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    nome_zip = PASTA_BACKUP / f"backup_{NOME_PROJETO}_{timestamp}.zip"
    
    if not silencioso:
        print(f"\n🚀 Iniciando backup de: {NOME_PROJETO}")
        print(f"📂 Destino: {nome_zip.name}")
        print(f"🚫 Ignorando: {', '.join(list(PASTAS_IGNORAR)[:4])}...")

    total_arquivos = 0
    arquivos_ignorados = 0
    
    with zipfile.ZipFile(nome_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PASTA_RAIZ):
            # [TRUQUE DE MESTRE] 
            # Modifica a lista 'dirs' in-place. O os.walk NÃO entrará nestas pastas.
            # Isso evita ler os milhares de arquivos do .venv
            dirs[:] = [d for d in dirs if d not in PASTAS_IGNORAR]
            
            for file in files:
                caminho_completo = Path(root) / file
                
                # Regras de exclusão de arquivos
                if (caminho_completo.suffix.lower() in EXTENSOES_IGNORAR or 
                    caminho_completo.name == SCRIPT_ATUAL.name or
                    caminho_completo == nome_zip):
                    arquivos_ignorados += 1
                    continue
                
                try:
                    # Caminho relativo para dentro do ZIP
                    arcname = caminho_completo.relative_to(PASTA_RAIZ)
                    zipf.write(caminho_completo, arcname)
                    total_arquivos += 1
                    
                    if not silencioso and total_arquivos % 50 == 0:
                        print(f"   ⏳ Processando... {total_arquivos} arquivos", end='\r')
                except Exception as e:
                    print(f"   ⚠️ Erro ao ler {file}: {e}")

    if not silencioso:
        tamanho_mb = nome_zip.stat().st_size / (1024 * 1024)
        print(f"\n✅ BACKUP CONCLUÍDO!")
        print(f"   📄 Arquivos: {total_arquivos}")
        print(f"   📦 Tamanho:  {tamanho_mb:.2f} MB")
        input("\nPress Enter...")
    
    return nome_zip

# ==============================================================================
# 2. 🔄 RESTORE SEGURO (CLEAN RESTORE)
# ==============================================================================
def restore_backup():
    # Busca ZIPs ordenados por data de modificação (mais recente primeiro)
    backups = sorted(PASTA_BACKUP.glob("*.zip"), key=os.path.getmtime, reverse=True)
    
    if not backups:
        print("\n⚠️  Nenhum backup encontrado.")
        input("Enter...")
        return

    print("\n📂 BACKUPS DISPONÍVEIS:")
    for i, b in enumerate(backups[:10], 1):
        data_mod = datetime.fromtimestamp(b.stat().st_mtime).strftime('%d/%m/%Y %H:%M')
        tamanho = b.stat().st_size / 1024 / 1024
        print(f"  {i}. {data_mod} — {b.name} ({tamanho:.2f} MB)")

    try:
        escolha = int(input("\n🔢 Digite o número (0 cancela): ")) - 1
        if escolha < 0: return
        zip_selecionado = backups[escolha]
    except (ValueError, IndexError):
        return

    print(f"\n⚠️  ATENÇÃO: O DIRETÓRIO ATUAL SERÁ LIMPO!")
    print(f"   Isso garante que arquivos antigos não se misturem com o backup.")
    confirm = input("🔒 Digite 'SIM' para confirmar: ").strip().upper()
    
    if confirm != 'SIM':
        print("❌ Operação cancelada.")
        input("Enter...")
        return

    # 1. Backup de Segurança Automático
    print("\n🛡️  Criando backup de segurança do estado atual...")
    backup_projeto(silencioso=True)

    # 2. Limpeza do Diretório (Preserva Backup e o Script)
    print("🧹 Limpando arquivos atuais...")
    for item in PASTA_RAIZ.iterdir():
        if item.name in {PASTA_BACKUP.name, SCRIPT_ATUAL.name, '.git', '.venv', 'venv'}:
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"   ! Falha ao deletar {item.name}: {e}")

    # 3. Extração
    print(f"📦 Extraindo {zip_selecionado.name}...")
    try:
        with zipfile.ZipFile(zip_selecionado, 'r') as z:
            z.extractall(PASTA_RAIZ)
        print("\n✅ RESTAURAÇÃO CONCLUÍDA COM SUCESSO!")
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NA EXTRAÇÃO: {e}")
        print("   O backup de segurança anterior foi criado na pasta backup.")
    
    input("Enter...")

# ==============================================================================
# 3. 📋 REQUIREMENTS INTELIGENTE (SCANNER DE IMPORTS)
# ==============================================================================
def criar_requirements():
    print(f"\n🔍 Analisando arquivos .py em busca de imports...")
    nome_arq = PASTA_BACKUP / f"requirements_{NOME_PROJETO}.txt"
    
    imports_encontrados = set()
    
    # Regex para capturar 'import xxx' ou 'from xxx'
    padrao_import = re.compile(r'^\s*(?:import|from)\s+(\w+)', re.MULTILINE)

    for root, dirs, files in os.walk(PASTA_RAIZ):
        dirs[:] = [d for d in dirs if d not in PASTAS_IGNORAR]
        
        for file in files:
            if file.endswith(".py"):
                caminho = Path(root) / file
                try:
                    conteudo = caminho.read_text(encoding='utf-8', errors='ignore')
                    matchs = padrao_import.findall(conteudo)
                    imports_encontrados.update(matchs)
                except Exception:
                    pass

    # Filtra módulos padrão do Python (os, sys, math, etc) para não poluir
    bibliotecas_padrao = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else {}
    imports_limpos = {imp for imp in imports_encontrados if imp not in bibliotecas_padrao}
    
    # Mapeamento de nomes de import -> nomes de pacote pip (Ex: cv2 -> opencv-python)
    # Adicione aqui correções comuns se necessário
    correcoes = {
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'yaml': 'PyYAML'
    }
    
    lista_final = []
    for imp in imports_limpos:
        pacote = correcoes.get(imp, imp)
        # Ignora imports locais (do próprio projeto)
        if not (PASTA_RAIZ / f"{imp}.py").exists() and not (PASTA_RAIZ / imp).is_dir():
            lista_final.append(pacote)

    with open(nome_arq, 'w', encoding='utf-8') as f:
        f.write(f"# Gerado em {datetime.now()}\n")
        f.write(f"# Projeto: {NOME_PROJETO}\n\n")
        for pacote in sorted(lista_final):
            f.write(f"{pacote}\n")

    print(f"✅ Requirements gerado: {nome_arq.name}")
    print(f"   Encontrados {len(lista_final)} pacotes externos prováveis.")
    input("Enter...")
    return nome_arq

# ==============================================================================
# 4. 🧠 GERADOR DE CONTEXTO PARA IA
# ==============================================================================
def gerar_documentacao_ia():
    nome_arq = PASTA_BACKUP / f"contexto_ia_{NOME_PROJETO}.txt"
    print(f"\n🤖 Gerando documentação unificada para IA...")
    
    # Extensões de texto legível
    extensoes_texto = {'.py', '.js', '.html', '.css', '.json', '.sql', '.md', '.txt', '.env'}
    
    with open(nome_arq, 'w', encoding='utf-8') as f_out:
        f_out.write(f"PROJETO: {NOME_PROJETO.upper()}\n")
        f_out.write(f"DATA: {datetime.now()}\n")
        f_out.write("="*80 + "\n\n")

        # 1. Estrutura de Diretórios
        f_out.write(">>> ESTRUTURA DE ARQUIVOS <<<\n")
        for root, dirs, files in os.walk(PASTA_RAIZ):
            dirs[:] = [d for d in dirs if d not in PASTAS_IGNORAR]
            nivel = root.replace(str(PASTA_RAIZ), '').count(os.sep)
            indentacao = ' ' * 4 * nivel
            f_out.write(f"{indentacao}{os.path.basename(root)}/\n")
            subindent = ' ' * 4 * (nivel + 1)
            for file in files:
                if Path(file).suffix in extensoes_texto:
                    f_out.write(f"{subindent}{file}\n")
        f_out.write("\n" + "="*80 + "\n\n")

        # 2. Conteúdo dos Arquivos
        f_out.write(">>> CONTEÚDO DOS ARQUIVOS <<<\n")
        contador = 0
        for root, dirs, files in os.walk(PASTA_RAIZ):
            dirs[:] = [d for d in dirs if d not in PASTAS_IGNORAR]
            
            for file in files:
                caminho = Path(root) / file
                if caminho.suffix in extensoes_texto and caminho.name != SCRIPT_ATUAL.name:
                    rel_path = caminho.relative_to(PASTA_RAIZ)
                    f_out.write(f"\n{'='*80}\nARQUIVO: {rel_path}\n{'='*80}\n")
                    
                    try:
                        # Limite de tamanho para não travar a IA (max 500KB por arquivo)
                        if caminho.stat().st_size > 500 * 1024:
                            f_out.write(f"[ARQUIVO MUITO GRANDE OMISTIDO: {caminho.stat().st_size} bytes]\n")
                        else:
                            f_out.write(caminho.read_text(encoding='utf-8', errors='ignore'))
                            contador += 1
                    except Exception as e:
                        f_out.write(f"[ERRO DE LEITURA: {e}]\n")

    print(f"✅ Contexto salvo: {nome_arq.name}")
    print(f"   Total de arquivos compilados: {contador}")
    input("Enter...")
    return nome_arq

# ==============================================================================
# 5. ⚡ MODO DEUS (TUDO DE UMA VEZ)
# ==============================================================================
def modo_deus():
    print("\n⚡ ATIVANDO MODO DEUS...")
    try:
        b = backup_projeto(silencioso=True)
        print(f"✅ Backup: OK ({b.name})")
        
        r = criar_requirements()
        print(f"✅ Requirements: OK")
        
        d = gerar_documentacao_ia()
        print(f"✅ Docs IA: OK")
        
        print("\n🎉 MODO DEUS FINALIZADO COM SUCESSO!")
    except Exception as e:
        print(f"\n❌ ERRO NO MODO DEUS: {e}")
    input("Enter...")

# ==============================================================================
# MENU PRINCIPAL
# ==============================================================================
def menu():
    while True:
        limpar_tela()
        print("="*60)
        print(f"   🛡️  GERENCIADOR SUPREMO: {NOME_PROJETO.upper()}")
        print("="*60)
        print(f"   📂 Local: {PASTA_RAIZ}")
        print("="*60)
        print("   1. 📦 Backup Completo (Ignora .venv/git)")
        print("   2. 🔄 Restaurar Backup (Modo Limpo)")
        print("   3. 📋 Gerar requirements.txt (Scanner)")
        print("   4. 🤖 Gerar Documentação para IA")
        print("   5. ⚡ MODO DEUS (Gerar Tudo)")
        print("   0. 🚪 Sair")
        print("="*60)
        
        op = input("\n   👉 Opção: ").strip()

        if op == "1": backup_projeto()
        elif op == "2": restore_backup()
        elif op == "3": criar_requirements()
        elif op == "4": gerar_documentacao_ia()
        elif op == "5": modo_deus()
        elif op == "0":
            print("\n👋 Até logo, Dr. Francisco Netto.")
            break
        else:
            print("❌ Opção inválida!")
            input("Enter...")

if __name__ == "__main__":
    menu()