import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd_calamari.ocr import CalamariOcr


@click.command()
@ocrd_cli_options
def ocrd_calamari_ocr(*args, **kwargs):
    return ocrd_cli_wrap_processor(CalamariOcr, *args, **kwargs)
