# Use Node.js LTS
FROM node:20-slim

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (use --only=production for smaller image if you don’t need dev deps)
RUN npm install --omit=dev

# Copy all project files
COPY . .

# Expose the port Cloud Run will use
ENV PORT=8080
EXPOSE 8080

# Start the app
CMD ["node", "server.js"]
