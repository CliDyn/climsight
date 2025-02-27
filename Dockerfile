FROM mambaorg/micromamba
# Add metadata
LABEL org.opencontainers.image.title="ClimSight"
LABEL org.opencontainers.image.description="A tool that combines LLMs with climate data to provide localized insights for decision-making in agriculture, urban planning, disaster management, and policy development."
LABEL org.opencontainers.image.authors="koldunovn, kuivi, AntoniaJost, dmpantiu, boryasbora"
LABEL org.opencontainers.image.url="https://github.com/CliDyn/climsight"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.source="https://github.com/CliDyn/climsight"
LABEL org.label-schema.citation="https://doi.org/10.1038/s43247-023-01199-1"

COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app
# Yeah, this next one is dumb. But it seems to be a requirement either in
# Docker or in Mamba, Paul can't tell which but this "does the trick":
RUN sed -i 's/name: climsight/name: base/g' ./environment.yml
RUN micromamba install -f ./environment.yml && \
  micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# Streamlit settings
ENV STREAMLIT_SERVER_PORT=8501

# Expose port
EXPOSE 8501

# Set an environment variable for optional arguments, default is empty
ENV STREAMLIT_ARGS=""

# Command to run Streamlit, using the environment variable
CMD streamlit run src/climsight/climsight.py $STREAMLIT_ARGS
