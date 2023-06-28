FROM python:3.10-slim

WORKDIR /app

ARG DB_USER
ARG DB_PASSWORD
ARG DB_HOST
ARG DB_PORT
ARG DB_NAME
ARG ANSIBLE_PASSWD
ARG KAFKAHOST
ARG KAFKAPORT
ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# save ansible password and other credentials
RUN echo $ANSIBLE_PASSWD >> ansible.credentials && \
    echo $DB_USER >> db.credentials && \
    echo $DB_PASSWORD >> db.credentials && \
    echo $DB_HOST >> db.credentials && \
    echo $DB_PORT >> db.credentials && \
    echo $DB_NAME >> db.credentials && \
    echo $KAFKAHOST >> kafka.credentials && \
    echo $KAFKAPORT >> kafka.credentials

RUN cat ansible.credentials && echo $ANSIBLE_PASSWD && echo test

RUN ansible-vault encrypt db.credentials --vault-password-file=ansible.credentials
RUN ansible-vault encrypt kafka.credentials --vault-password-file=ansible.credentials

COPY . .
