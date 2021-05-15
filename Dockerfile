FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/joe/github/final_year_project_jarrow/

COPY requirements.txt requirements.txt

# Install tensorflow and reqs
RUN pip install -U pip
RUN pip install -U tensorflow-addons
RUN pip install sklearn
RUN pip install pandas
RUN pip install pytest
RUN pip install matplotlib
RUN pip install flake8
RUN pip install jupyterlab
RUN pip install future-breakpoint
RUN pip install random_name

RUN apt-get update
RUN apt-get install -y tmux
RUN apt-get install -y nano
RUN apt-get install -y sudo
RUN apt-get install -y task-spooler
RUN apt-get install -y less
RUN apt-get install -y git

ARG user=joe
ARG group=joe
ARG uid=1000
ARG gid=1000

RUN groupadd -g ${gid} ${group} && useradd -u ${uid} -g ${group} -s /bin/sh ${user}
RUN usermod -a -G sudo joe
RUN echo -e "linux\nlinux" | passwd joe
RUN mkdir -p /home/joe/.local/share/nano

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
