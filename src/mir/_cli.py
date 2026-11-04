"""Console entry point; the command parser owns early numerical configuration."""
def main():
    from .cli import main as run

    run()
