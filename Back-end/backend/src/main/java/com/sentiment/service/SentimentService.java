package com.sentiment.service;

import com.sentiment.model.SentimentResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

@Service
public class SentimentService {

    private static final String PYTHON_SERVICE_URL = "http://localhost:5000";
    private final RestTemplate restTemplate;

    public SentimentService() {
        this.restTemplate = new RestTemplate();
    }

    public SentimentResponse analyzeSentiment(String text) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.TEXT_PLAIN);

        HttpEntity<String> request = new HttpEntity<>(text, headers);
        
        try {
            ResponseEntity<Map> response = restTemplate.postForEntity(
                PYTHON_SERVICE_URL + "/analyze",
                request,
                Map.class
            );

            Map<String, Object> responseBody = response.getBody();
            
            if (responseBody != null) {
                return new SentimentResponse(
                    (String) responseBody.get("text"),
                    (String) responseBody.get("sentiment"),
                    ((Number) responseBody.get("score")).doubleValue(),
                    ((Number) responseBody.get("score")).intValue()
                );
            }
            
            throw new RuntimeException("Python servisinden geçersiz yanıt alındı");
        } catch (Exception e) {
            throw new RuntimeException("Duygu analizi sırasında bir hata oluştu: " + e.getMessage());
        }
    }

    public List<SentimentResponse> getRecentAnalyses() {
        try {
            ResponseEntity<List> response = restTemplate.getForEntity(
                PYTHON_SERVICE_URL + "/recent",
                List.class
            );

            List<Map<String, Object>> analyses = response.getBody();
            
            List<SentimentResponse> results = new ArrayList<>();
            if (analyses != null) {
                for (Map<String, Object> analysis : analyses) {
                    results.add(new SentimentResponse(
                        (String) analysis.get("text"),
                        (String) analysis.get("sentiment"),
                        ((Number) analysis.get("score")).doubleValue(),
                        ((Number) analysis.get("score")).intValue()
                    ));
                }
            }
            
            return results;
        } catch (Exception e) {
            throw new RuntimeException("Son analizleri alırken bir hata oluştu: " + e.getMessage());
        }
    }
} 