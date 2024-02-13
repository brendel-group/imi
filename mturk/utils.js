function fetchAllJson(...resources) {
  var destination = [];
  resources.forEach((it) => {
    destination.push(
      fetch(it, { cache: "no-store" }).then((response) => {
        if (response.status !== 200) {
          throw new Error("fetch returned status " + response.status, it);
        } else {
          try {
            return response.json();
          } catch (e) {
            throw new Error(e.message + "file: " + it);
          }
        }
      })
    );
  });
  return Promise.all(destination);
}

function fetchJson(resource, tries = 60, delay= 1000) {
  return fetchRetry(resource, delay, tries, { cache: "no-store" }
  ).then((response) => {
    if (response.status !== 200) {
      throw new Error("fetch returned status " + response.status, it);
    } else {
      try {
        return response.json();
      } catch (e) {
        throw new Error(e.message + "file: " + it);
      }
    }
  });
}

function wait(delay){
    return new Promise((resolve) => setTimeout(resolve, delay));
}

function fetchRetry(url, delay, tries, fetchOptions = {}) {
    function onError(err){
        triesLeft = tries - 1;
        if(!triesLeft){
            throw err;
        }
        return wait(delay).then(() => fetchRetry(url, delay, triesLeft, fetchOptions));
    }
    return fetch(url,fetchOptions).catch(onError);
}

async function fetchWithTimeout(resource, options) {
  const { timeout = 8000 } = options;

  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);

  const response = await fetch(resource, {
    ...options,
    signal: controller.signal,
  });
  clearTimeout(id);

  return response;
}

/*
Returns all possible permutations of on array
*/
function permutator(inputArr) {
  var results = [];

  function permute(arr, memo) {
    var cur,
      memo = memo || [];

    for (var i = 0; i < arr.length; i++) {
      cur = arr.splice(i, 1);
      if (arr.length === 0) {
        results.push(memo.concat(cur));
      }
      permute(arr.slice(), memo.concat(cur));
      arr.splice(i, 0, cur[0]);
    }

    return results;
  }

  return permute([...inputArr]);
}

/*
Returns a random subset of an array.
 */
function getRandomSubset(arr, size) {
  return Array.from({length: size}, () => getRandomInt(0, arr.length - 1)
  ).map((it) => arr[it]);
}

/*
 * Returns a random integer between min (inclusive) and max (inclusive).
 * The value is no lower than min (or the next integer greater than min
 * if min isn't an integer) and no greater than max (or the next integer
 * lower than max if max isn't an integer).
 * Using Math.round() will give you a non-uniform distribution!
 */
function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

if (!Array.prototype.last) {
  Array.prototype.last = function () {
    return this[this.length - 1];
  };
}


if (!Array.prototype.shuffle) {
  Array.prototype.shuffle = function () {
    let currentIndex = this.length,  randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {
      // Pick a remaining element.
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;

      // And swap it with the current element.
      [this[currentIndex], this[randomIndex]] = [
        this[randomIndex], this[currentIndex]];
    }

    return this;
  };
}