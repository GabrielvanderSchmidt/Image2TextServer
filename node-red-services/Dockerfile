FROM nodered/node-red

RUN npm install node-red-contrib-cloudantplus node-red-node-watson

ADD flows.json flows.json

CMD ["node-red", "flows.json"]
