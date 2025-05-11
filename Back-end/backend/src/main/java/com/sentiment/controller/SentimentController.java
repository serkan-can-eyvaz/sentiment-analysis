package com.sentiment.controller;

import com.sentiment.model.SentimentResponse;
import com.sentiment.service.SentimentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/sentiment")
@CrossOrigin(origins = "http://localhost:3000")
public class SentimentController {

    private final SentimentService sentimentService;

    @Autowired
    public SentimentController(SentimentService sentimentService) {
        this.sentimentService = sentimentService;
    }

    @PostMapping("/analyze")
    public ResponseEntity<SentimentResponse> analyzeSentiment(@RequestBody String text) {
        SentimentResponse result = sentimentService.analyzeSentiment(text);
        return ResponseEntity.ok(result);
    }
    
    @GetMapping("/recent")
    public ResponseEntity<List<SentimentResponse>> getRecentAnalyses() {
        List<SentimentResponse> results = sentimentService.getRecentAnalyses();
        return ResponseEntity.ok(results);
    }
} 