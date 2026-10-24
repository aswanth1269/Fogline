import os
from backend.app import app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5000'))
    print(f'Open http://127.0.0.1:{port}')
    app.run(host='127.0.0.1', port=port, debug=False)
