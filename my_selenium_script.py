# this code was done for TESTING the website url
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

driver.get('https://u.ae/en/information-and-services')
html = driver.page_source
print(html)  # See if the data is now present

driver.quit()
