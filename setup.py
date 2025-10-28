#!/usr/bin/env python3
"""
Complete setup script for MedIntel RAG Chatbot
Run this script to set up everything automatically
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"📝 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        "data/raw_documents",
        "data/vector_store",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")


def create_env_file():
    """Create .env file from example"""
    print_header("Setting Up Environment")
    
    if Path(".env").exists():
        print("⚠️  .env file already exists, skipping...")
        return True
    
    if Path(".env.example").exists():
        with open(".env.example", "r") as src, open(".env", "w") as dst:
            content = src.read()
            dst.write(content)
        print("✅ Created .env file from .env.example")
        print("\n⚠️  IMPORTANT: Edit .env and add your API keys!")
        print("   nano .env")
        return True
    else:
        print("❌ .env.example not found")
        return False


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages"
    )


def ingest_sample_data():
    """Ingest sample medical data"""
    print_header("Ingesting Sample Data")
    
    return run_command(
        f"{sys.executable} scripts/ingest_data.py --source sample",
        "Creating sample medical data and vector store"
    )


def test_api():
    """Start API server in test mode"""
    print_header("Setup Complete!")
    
    print("🎉 MedIntel is ready to use!\n")
    print("Next steps:")
    print("1. Edit .env file and add your OpenAI API key (or set LLM_PROVIDER=mistral)")
    print("   nano .env")
    print("\n2. Start the API server:")
    print("   python -m src.api")
    print("\n3. In a new terminal, test the API:")
    print("   python scripts/test_api.py")
    print("\n4. Or open in browser:")
    print("   http://localhost:8000/docs")
    print("\n5. Deploy to production:")
    print("   See QUICKSTART.md for deployment instructions")
    print("\n" + "="*60)


def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("  🏥 MedIntel RAG Chatbot - Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    print("\n⏳ This may take a few minutes...")
    if not install_dependencies():
        print("\n❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Ingest sample data
    if not ingest_sample_data():
        print("\n⚠️  Failed to ingest sample data")
        print("   You can run it manually later:")
        print("   python scripts/ingest_data.py --source sample")
    
    # Final instructions
    test_api()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
