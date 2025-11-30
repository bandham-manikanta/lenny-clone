"""
CLI client for chatting with Lenny
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from rag import LennyRAG
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def print_header():
    """Print welcome header"""
    header = """
    🎙️  Chat with Lenny Rachitsky
    
    Ask me anything about product management, growth, and building products.
    Type 'quit' or 'exit' to end the conversation.
    """
    console.print(Panel(header, style="bold blue"))


def main():
    """Main CLI loop"""
    
    print_header()
    
    # Initialize RAG
    console.print("\n[yellow]Initializing...[/yellow]")
    try:
        rag = LennyRAG()
        console.print("[green]✅ Ready![/green]\n")
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize: {e}[/red]")
        return
    
    # Chat loop
    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if question.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]Thanks for chatting! 👋[/yellow]\n")
                break
            
            if not question.strip():
                continue
            
            # Get response with streaming
            console.print("\n[bold green]Lenny[/bold green]: ", end="")
            
            full_response = ""
            for chunk in rag.query(question, stream=True, top_k=5):
                console.print(chunk, end="")
                full_response += chunk
            
            console.print("\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]\n")
            break
        
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    # Install rich if needed
    try:
        from rich.console import Console
    except ImportError:
        print("Installing required package: rich")
        os.system("uv pip install rich")
        from rich.console import Console
    
    main()
