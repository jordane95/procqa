
sudo apt update
sudo apt install apt-transport-https

sudo apt install openjdk-8-jdk

java -version

wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

sudo sh -c 'echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" > /etc/apt/sources.list.d/elastic-7.x.list'

sudo apt update
sudo apt install elasticsearch

sudo systemctl enable elasticsearch.service
sudo systemctl start elasticsearch.service

curl -X GET "localhost:9200/"
