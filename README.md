# README

### Deploy on Google Cloud Run

```sh
docker build -t gcr.io/YOUR_PROJECT_ID/YOUR_IMAGE_NAME .
```

```sh
docker push gcr.io/YOUR_PROJECT_ID/YOUR_IMAGE_NAME
```

```sh
gcloud run deploy --image gcr.io/YOUR_PROJECT_ID/YOUR_IMAGE_NAME --platform managed
```
