Bouncer is a flask app that checks that no MTurk worker participates more than once in an experiment.
Before MTurk workers have the chance to work on a HIT our UI has to check whether the bouncer lets them pass to the experiment.
Once they have finished the experiment, our UI has to send a request back to bouncer to ban them from further participation.
Previous participation will be logged in an SQLLite3 database.

# Good to Know

## On Roland's Server
This is where the databases are located: `interpretability_comparison/tools/mturk/mturk-bouncer/data`

The one where the real ICLR replication data is in is called `database.db.real`. It is also backed up as `database.db.backup.20212804`.

The one that is always used is called `database.db`.


## Workflow

### For Testing in Sandbox
When testing and re-allowing yourself in the sandbox, execute
`sudo rm database.db`.

To restart the database (i.e. create a new database), execute
`docker-compose -f ~/interpretability_comparison/server/web-data/docker-compose.yml restart bouncer`
and wait (quite a while) until you see `Restarting webdata_bouncer_1 ... done`.


### For Real
Copy the `database.db.real` over to `database.db` so that the previous workers are excluded.

Then run the experiment.

Finally, make a backup of `database.db` as well as a local copy.

