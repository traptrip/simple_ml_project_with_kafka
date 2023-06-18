FROM python:3.10-slim

WORKDIR /app

ARG DB_USER
ARG DB_PASSWORD
ARG DB_HOST
ARG DB_PORT
ARG DB_NAME
ARG ANSIBLE_PASSWD

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# save ansible password and other credentials
RUN touch ansible.credentials && \
    echo $ANSIBLE_PASSWD >> ansible.credentials && \
    touch db.credentials && \
    echo $DB_USER >> db.credentials && \
    echo $DB_PASSWORD >> db.credentials && \
    echo $DB_HOST >> db.credentials && \
    echo $DB_PORT >> db.credentials && \
    echo $DB_NAME >> db.credentials && \
    ansible-vault encrypt db.credentials --vault-password-file=ansible.credentials

COPY . .
