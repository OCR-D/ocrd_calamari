ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE
ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/OCR-D/ocrd_calamari" \
    org.label-schema.build-date=$BUILD_DATE \
    org.opencontainers.image.vendor="DFG-Funded Initiative for Optical Character Recognition Development" \
    org.opencontainers.image.title="ocrd_calamari" \
    org.opencontainers.image.description="OCR-D compliant workspace processor for the functionality of Calamari OCR" \
    org.opencontainers.image.source="https://github.com/OCR-D/ocrd_calamari" \
    org.opencontainers.image.documentation="https://github.com/OCR-D/ocrd_calamari/blob/${VCS_REF}/README.md" \
    org.opencontainers.image.revision=$VCS_REF \
    org.opencontainers.image.created=$BUILD_DATE \
    org.opencontainers.image.base.name=$DOCKER_BASE_IMAGE
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONIOENCODING utf8
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /build/calamari
COPY Makefile .
COPY pyproject.toml .
COPY ocrd-tool.json .
COPY requirements.txt .
COPY README.md .
COPY ocrd_calamari ./ocrd_calamari
RUN make install && \
    rm -rf /build/calamari

RUN ocrd resmgr download -l system ocrd-calamari-recognize qurator-gt4histocr-1.0 && \
    rm -rf /tmp/ocrd_*

WORKDIR /data
VOLUME ["/data"]
