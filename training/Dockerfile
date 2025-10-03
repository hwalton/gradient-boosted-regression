FROM public.ecr.aws/lambda/python:3.10

# Build tools for wheels
RUN yum install -y git curl bzip2 tar gcc gcc-c++ make cmake && yum clean all

# Copy container-only requirements so layer does not include pandas
COPY requirements.container.txt /tmp/requirements.txt

# Use the Lambda runtime python for pip, install into /opt/python (Lambda layer path)
RUN set -eux; \
    /var/lang/bin/python -m pip install --upgrade pip setuptools wheel; \
    /var/lang/bin/python -m pip install --no-cache-dir -r /tmp/requirements.txt -t /opt/python; \
    rm -rf /root/.cache/pip /tmp/*

# Ensure runtime can find packages and runtime python is on PATH
ENV PYTHONPATH=/opt/python:${PYTHONPATH:-}
ENV PATH=/var/lang/bin:${PATH}

# Copy app code
COPY . /var/task
WORKDIR /var/task

# Build-time sanity check: fail build if a key package isn't importable
RUN /var/lang/bin/python -c "import importlib,sys; importlib.invalidate_caches(); import pandas; print('pandas', pandas.__version__)"

CMD ["training.lambda_train.handler"]