package com.sentiment.model;

public class SentimentResponse {
    private String text;
    private String sentiment;
    private double score;
    private int stars;

    public SentimentResponse(String text, String sentiment, double score, int stars) {
        this.text = text;
        this.sentiment = sentiment;
        this.score = score;
        this.stars = stars;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public String getSentiment() {
        return sentiment;
    }

    public void setSentiment(String sentiment) {
        this.sentiment = sentiment;
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    public int getStars() {
        return stars;
    }

    public void setStars(int stars) {
        this.stars = stars;
    }
} 