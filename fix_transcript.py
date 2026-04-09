import re

# Read the transcript
input_path = r"c:\Users\halfo\OneDrive\Documents\GitHub\whisperX-medical-transcribe\uploads\test4-3 - Copy - Copy - Copy - Copy - Copy.md"
output_path = r"c:\Users\halfo\OneDrive\Documents\GitHub\whisperX-medical-transcribe\uploads\test4-3-fixed.md"

with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()

# Parse lines
pattern = re.compile(r'\*\*\[(\d+:\d+(?::\d+)?)\] (SPEAKER_\d+):\*\* (.+)')
lines = content.strip().split('\n')

segments = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    
    match = pattern.match(line)
    if match:
        timestamp, speaker, text = match.groups()
        segments.append({
            "timestamp": timestamp,
            "speaker": speaker,
            "text": text.strip()
        })

# Speaker mapping as specified by user:
# SPEAKER_00 = Интервьюер, SPEAKER_01 = Респондент
speaker_map = {
    "SPEAKER_00": "Интервьюер",
    "SPEAKER_01": "Респондент"
}

# Merge consecutive segments from same speaker
merged = []
current = None

for seg in segments:
    speaker_name = speaker_map.get(seg["speaker"], seg["speaker"])
    
    if current is None:
        current = {
            "timestamp": seg["timestamp"],
            "speaker": speaker_name,
            "text": seg["text"]
        }
    elif speaker_name == current["speaker"]:
        # Same speaker - merge text
        current["text"] += " " + seg["text"]
    else:
        # Different speaker - save previous and start new
        merged.append(current)
        current = {
            "timestamp": seg["timestamp"],
            "speaker": speaker_name,
            "text": seg["text"]
        }

if current:
    merged.append(current)

# Generate output
output_lines = []
for seg in merged:
    output_lines.append(f"**[{seg['timestamp']}] {seg['speaker']}:** {seg['text']}")

output_content = "\n\n".join(output_lines)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_content)

print(f"Processed {len(segments)} segments into {len(merged)} merged blocks")
print(f"Output saved to: {output_path}")
