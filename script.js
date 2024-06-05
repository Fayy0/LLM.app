const queryInput = document.getElementById('query');
const searchForm = document.getElementById('search-form');
const results = document.getElementById('results');
const error = document.getElementById('error');

searchForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  if (!query) {
    error.textContent = 'Please enter a search term.';
    return;
  }

  results.innerHTML = '';
  error.textContent = '';

  const eventSource = new EventSource(`http://127.0.0.1:5000/search_stream?query=${encodeURIComponent(query)}`);

  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.error) {
      error.textContent = `Error: ${data.error}`;
      eventSource.close();
    } else {
      results.innerHTML += `<p><strong>${data.model}</strong>: ${data.response}<br>
                            Response Time: ${data.response_time.toFixed(2)}s<br>
                            Relevance Score: ${data.relevance_score.toFixed(2)}<br>
                            Accuracy Score: ${data.accuracy_score.toFixed(2)}</p>`;
    }
  };

  eventSource.onerror = function() {
    console.error('EventSource failed');
    eventSource.close();
  };
});
