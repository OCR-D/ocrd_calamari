import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_calamari.recognize import CalamariRecognize


@click.command()
@ocrd_cli_options
def ocrd_calamari_recognize(*args, **kwargs):
    """
    Run Calamari OCR multi-model recognition and voting
    """
    return ocrd_cli_wrap_processor(CalamariRecognize, *args, **kwargs)
