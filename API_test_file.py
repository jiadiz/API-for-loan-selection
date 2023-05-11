#!/usr/bin/env python
# coding: utf-8

# In[16]:


from fastapi.testclient import TestClient
import io
import csv
from python_file_for_loan_api import predict_default,loan_default  # Import the new function


client = TestClient(loan_default)  # Update this line accordingly

def test_loan_default(mocker):
    # Arrange
    filename = "APItest2.csv"

    with open(filename, "rb") as csvfile:
        file_content = csvfile.read()

    mocker.patch("loan_api.predict_default", return_value=["loan12345 0.0567", "loan12346 0.0250", "loan12347 0.0756"])

    # Act
    response = client.post(
        "/predict",
        files={"file": ("APItest2.csv", file_content, "text/csv")}
    )

    # Assert
    assert response.status_code == 200
    assert response.json() == {"predictions": ["loan12345 0.0567", "loan12346 0.0250", "loan12347 0.0756"]}

