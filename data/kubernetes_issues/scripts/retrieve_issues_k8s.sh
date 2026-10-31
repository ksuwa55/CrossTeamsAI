cd "$(dirname "$0")/.."

for id in 109553 129652 115362 103587 113627; do
    gh issue view "$id" \
        --repo kubernetes/kubernetes \
        --json number,title,body,state,createdAt,closedAt,author,labels,comments,url \
        > "raw/issue_${id}.json"
done