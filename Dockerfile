FROM tensorflow/tensorflow:latest-gpu

ADD . /augpoliciesrepo

RUN cd augpoliciesrepo
RUN python -m venv env
RUN source env/bin/activate
RUN python -m pip install -U pip
RUN python -m pip install -r reqs.txt -y

LABEL maintainer="joearrowsmith98@gmail.com"
