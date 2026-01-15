package providers

import "fmt"

// wraps non-2xx responses from provider APIs.
type APIError struct {
	StatusCode int
	Body       string
}

func (err APIError) Error() string {
	return fmt.Sprintf("request failed with status %d: %s", err.StatusCode, err.Body)
}
