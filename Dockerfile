FROM mambaorg/micromamba


COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app
# Yeah, this next one is dumb. But it seems to be a requirement either in
# Docker or in Mamba, Paul can't tell which but this "does the trick":
RUN sed -i 's/name: climsight/name: base/g' ./environment.yml
RUN micromamba install -f ./environment.yml && \
  micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN python download_data.py

WORKDIR /app
# Streamlit settings
ENV STREAMLIT_SERVER_PORT=8501

# Expose port
EXPOSE 8501

# Set an environment variable for optional arguments, default is empty
ENV STREAMLIT_ARGS=""

# Command to run Streamlit, using the environment variable
CMD streamlit run src/climsight/climsight.py $STREAMLIT_ARGS
