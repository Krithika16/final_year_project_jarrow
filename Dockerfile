FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/joe/github/final_year_project_jarrow/

COPY requirements.txt requirements.txt

# Install tensorflow and reqs
RUN pip install -U pip
RUN pip install -U tensorflow
RUN pip install -U tensorflow-addons
RUN pip install sklearn
RUN pip install pandas
RUN pip install pytest
RUN pip install matplotlib
RUN pip install flake8

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
