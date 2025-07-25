#!/bin/bash
echo "Deploying House Price Predictor..."
git init
git add .
git commit -m "Initial commit"
heroku create house-price-fastapi
heroku git:remote -a house-price-fastapi
git push heroku master
