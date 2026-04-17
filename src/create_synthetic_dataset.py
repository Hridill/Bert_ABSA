import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_synthetic_dataset(num_samples_per_class=50000):
    # Define positive healthcare reviews with more variety
    positive_reviews = [
        "The doctor was very professional and caring.",
        "Excellent medical care and attention to detail.",
        "The hospital staff was very helpful and friendly.",
        "Great experience with the medical team.",
        "The treatment was effective and the recovery was quick.",
        "The doctor explained everything clearly and was very patient.",
        "The hospital facilities are modern and clean.",
        "The medical staff was very knowledgeable and experienced.",
        "The appointment was on time and the service was efficient.",
        "The doctor's diagnosis was accurate and the treatment worked well.",
        "The nurse was attentive and made me feel comfortable.",
        "The specialist provided excellent care and follow-up.",
        "The emergency room staff handled my case promptly.",
        "The pediatrician was great with my children.",
        "The physical therapy sessions were very effective.",
        "The hospital's cleanliness standards were impressive.",
        "The doctor's bedside manner was exceptional.",
        "The medical team's coordination was seamless.",
        "The follow-up care was thorough and helpful.",
        "The hospital's technology and equipment were state-of-the-art."
    ]

    # Define neutral healthcare reviews with more variety
    neutral_reviews = [
        "The appointment was scheduled as planned.",
        "The hospital was busy but manageable.",
        "The doctor's office was located in a convenient area.",
        "The medical staff followed standard procedures.",
        "The waiting time was average for a busy clinic.",
        "The hospital has all the necessary equipment.",
        "The medical records were properly maintained.",
        "The appointment system works as expected.",
        "The hospital follows standard protocols.",
        "The medical staff was present during the visit.",
        "The check-in process was straightforward.",
        "The hospital parking was adequate.",
        "The medical forms were standard and clear.",
        "The waiting room had basic amenities.",
        "The appointment duration was as expected.",
        "The hospital's location was accessible.",
        "The medical staff wore proper uniforms.",
        "The hospital had standard visiting hours.",
        "The medical equipment appeared functional.",
        "The hospital's signage was clear and helpful."
    ]

    # Define negative healthcare reviews with more variety
    negative_reviews = [
        "The doctor was rude and unprofessional.",
        "The waiting time was extremely long.",
        "The medical staff seemed disorganized.",
        "The treatment was ineffective and expensive.",
        "The hospital facilities were outdated and dirty.",
        "The doctor didn't explain the diagnosis properly.",
        "The medical staff was unresponsive to concerns.",
        "The appointment was delayed without proper notice.",
        "The hospital billing system was confusing.",
        "The medical care was substandard and disappointing.",
        "The emergency room wait was unacceptable.",
        "The doctor seemed rushed and inattentive.",
        "The hospital staff was understaffed and overwhelmed.",
        "The medical equipment was outdated and unreliable.",
        "The follow-up care was non-existent.",
        "The hospital's communication was poor.",
        "The medical records were lost multiple times.",
        "The treatment plan was unclear and inconsistent.",
        "The hospital's cleanliness was concerning.",
        "The medical staff seemed inexperienced and unsure."
    ]

    # Create lists for data
    reviews = []
    sentiments = []
    dates = []

    # Generate random dates within the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Function to generate variations of a review
    def generate_variations(base_review, num_variations):
        variations = [base_review]
        words = base_review.split()
        
        for _ in range(num_variations - 1):
            # Randomly modify the review
            if random.random() < 0.3:  # 30% chance to add an adjective
                adj = random.choice(['very', 'really', 'quite', 'extremely', 'particularly'])
                words.insert(random.randint(0, len(words)), adj)
            if random.random() < 0.2:  # 20% chance to add a phrase
                phrases = ['in my opinion,', 'from my experience,', 'I found that', 'I noticed that']
                words.insert(0, random.choice(phrases))
            if random.random() < 0.2:  # 20% chance to modify punctuation
                words[-1] = words[-1].replace('.', random.choice(['!', '...', '.']))
            
            variations.append(' '.join(words))
        return variations

    # Add positive reviews
    print("Generating positive reviews...")
    for base_review in positive_reviews:
        variations = generate_variations(base_review, num_samples_per_class // len(positive_reviews))
        reviews.extend(variations)
        sentiments.extend(['positive'] * len(variations))
        dates.extend([start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(len(variations))])

    # Add neutral reviews
    print("Generating neutral reviews...")
    for base_review in neutral_reviews:
        variations = generate_variations(base_review, num_samples_per_class // len(neutral_reviews))
        reviews.extend(variations)
        sentiments.extend(['neutral'] * len(variations))
        dates.extend([start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(len(variations))])

    # Add negative reviews
    print("Generating negative reviews...")
    for base_review in negative_reviews:
        variations = generate_variations(base_review, num_samples_per_class // len(negative_reviews))
        reviews.extend(variations)
        sentiments.extend(['negative'] * len(variations))
        dates.extend([start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(len(variations))])

    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame({
        'review': reviews,
        'sentiment': sentiments,
        'date': dates
    })

    # Shuffle the dataset
    print("Shuffling dataset...")
    df = df.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    print("Saving to CSV...")
    df.to_csv('Input/healthcare_sentiment_dataset.csv', index=False)
    print(f"Created synthetic dataset with {len(df)} samples")
    print(f"Class distribution:")
    print(df['sentiment'].value_counts())
    print("Dataset saved to 'Input/healthcare_sentiment_dataset.csv'")

if __name__ == "__main__":
    create_synthetic_dataset() 