import argparse
from app.database import SessionLocal
from app.models import Recording
from app.services.text_processor import TextProcessingService

def main():
    parser = argparse.ArgumentParser(description='Backfill recording translations and intelligence.')
    parser.add_argument('--recording-id', type=str, default=None, help='Recording UUID to backfill')
    parser.add_argument('--all', action='store_true', help='Backfill all recordings in the database')
    parser.add_argument('--regenerate-intel', action='store_true', help='Regenerate study notes, assignments, quizzes')

    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.all:
            recordings = db.query(Recording).all()
            target_ids = [r.id for r in recordings]
        elif args.recording_id:
            target_ids = [args.recording_id]
        else:
            recordings = db.query(Recording).all()
            target_ids = [r.id for r in recordings]

        if not target_ids:
            print('No recordings found to backfill.')
            return

        print(f'Starting backfill for {len(target_ids)} recording(s). Regenerate intelligence: {args.regenerate_intel}')
        for rid in target_ids:
            print(f'\nProcessing recording: {rid}...')
            res = TextProcessingService.backfill_recording(rid, db, regenerate_intelligence=args.regenerate_intel)
            print(f"Successfully backfilled recording {rid}:")
            print(f"  - Total segments: {res.get('total_segments')}")
            print(f"  - English words: {res.get('english_transcript_words')}")
            print(f"  - Intelligence regenerated: {res.get('intelligence_regenerated')}")
    finally:
        db.close()


if __name__ == '__main__':
    main()
