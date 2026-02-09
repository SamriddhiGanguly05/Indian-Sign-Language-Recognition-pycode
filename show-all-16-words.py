# Display all 16 words the CRNN model can recognize
# This shows the complete vocabulary

classes = ['Loud','They','Sad','Quiet','He','Thank you','How are you','You','It','Good Afternoon','Hello','Alright','Beautiful','Happy','None','Good Morning']

print("="*70)
print("📚 COMPLETE VOCABULARY - 16 WORDS")
print("="*70)
print(f"\nTotal Words: {len(classes)}\n")

for i, word in enumerate(classes, 1):
    print(f"   {i:2d}. {word}")

print("\n" + "="*70)
print("📊 WORD CATEGORIES:")
print("="*70)
print("\nGreetings:")
greetings = ['Hello', 'Good Morning', 'Good Afternoon', 'How are you']
for word in greetings:
    if word in classes:
        print(f"   • {word}")

print("\nEmotions:")
emotions = ['Happy', 'Sad']
for word in emotions:
    if word in classes:
        print(f"   • {word}")

print("\nDescriptions:")
descriptions = ['Beautiful', 'Loud', 'Quiet', 'Alright']
for word in descriptions:
    if word in classes:
        print(f"   • {word}")

print("\nPronouns:")
pronouns = ['He', 'They', 'You', 'It']
for word in pronouns:
    if word in classes:
        print(f"   • {word}")

print("\nOther:")
other = ['Thank you', 'None']
for word in other:
    if word in classes:
        print(f"   • {word}")

print("\n" + "="*70)
print("✅ Model Status:")
print("="*70)
print("   • CRNN Model Architecture: ✅ Ready")
print("   • Model Weights: ⚠️  Need trained checkpoint file")
print("   • Pipeline (OpenCV): ✅ Working")
print("   • All 16 Words Defined: ✅ Ready")
print("="*70)

